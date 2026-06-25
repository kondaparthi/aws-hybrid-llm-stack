"""
inference_proxy/handler.py

Single entry point for all LLM inference requests.
Tries local Ollama first. Falls back to Bedrock on timeout or error.

Flow:
  1. Check if Ollama ECS service is running (scale up if needed)
  2. Call Ollama with configured timeout
  3. On timeout/error → Bedrock fallback (if enabled)
  4. Emit CloudWatch metrics (local vs fallback, latency, tokens)
"""

import os
import json
import time
import logging
import boto3
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ecs = boto3.client("ecs")
bedrock_runtime = boto3.client("bedrock-runtime")
ssm = boto3.client("ssm")
cloudwatch = boto3.client("cloudwatch")

ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")
OLLAMA_ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "")
ECS_CLUSTER = os.environ.get("ECS_CLUSTER", "")
ECS_SERVICE = os.environ.get("ECS_SERVICE", "")
SSM_PREFIX = os.environ.get("SSM_PREFIX", f"/aws-hybrid-llm/{ENVIRONMENT}")

_config_cache = {}


def get_config() -> dict:
    global _config_cache
    if _config_cache:
        return _config_cache
    try:
        response = ssm.get_parameters_by_path(Path=SSM_PREFIX, Recursive=True)
        params = {p["Name"].replace(f"{SSM_PREFIX}/", ""): p["Value"]
                  for p in response["Parameters"]}
        _config_cache = {
            "default_model": params.get("default-model", "llama3.2"),
            "bedrock_fallback_model": params.get("bedrock-fallback-model",
                                                  "anthropic.claude-3-haiku-20240307-v1:0"),
            "fallback_enabled": params.get("bedrock-fallback-enabled", "true") == "true",
            "local_timeout_ms": int(params.get("local-timeout-ms", "25000")),
        }
    except Exception as e:
        logger.warning(f"SSM load failed, using defaults: {e}")
        _config_cache = {
            "default_model": "llama3.2",
            "bedrock_fallback_model": "anthropic.claude-3-haiku-20240307-v1:0",
            "fallback_enabled": True,
            "local_timeout_ms": 25000,
        }
    return _config_cache


def ensure_ecs_running() -> bool:
    """
    Checks ECS service desired count.
    If 0 (scaled to zero), scales up to 1 and waits for healthy task.
    Returns True if service is ready, False if still warming up.
    """
    if not ECS_CLUSTER or not ECS_SERVICE:
        return True

    try:
        response = ecs.describe_services(
            cluster=ECS_CLUSTER,
            services=[ECS_SERVICE]
        )
        service = response["services"][0]
        desired = service["desiredCount"]
        running = service["runningCount"]

        if desired == 0:
            logger.info("ECS service at 0 — scaling up")
            ecs.update_service(
                cluster=ECS_CLUSTER,
                service=ECS_SERVICE,
                desiredCount=1
            )
            return False  # Cold start — caller should fallback to Bedrock this time

        return running > 0

    except Exception as e:
        logger.warning(f"ECS status check failed: {e}")
        return False


def call_ollama(prompt: str, model: str, context_chunks: list,
                timeout_ms: int) -> dict:
    """
    Calls Ollama /api/chat endpoint.
    Raises TimeoutError or urllib.error.URLError on failure.
    """
    messages = []
    if context_chunks:
        context = "\n\n".join(context_chunks)
        messages.append({
            "role": "system",
            "content": f"Answer using only this context:\n\n{context}"
        })
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": 1024,
            "temperature": 0.7
        }
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_ENDPOINT}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    timeout_sec = timeout_ms / 1000
    start = time.time()

    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        result = json.loads(resp.read())

    latency_ms = int((time.time() - start) * 1000)

    return {
        "answer": result.get("message", {}).get("content", ""),
        "model": f"local/{model}",
        "input_tokens": result.get("prompt_eval_count", 0),
        "output_tokens": result.get("eval_count", 0),
        "latency_ms": latency_ms,
        "source": "local"
    }


def call_bedrock(prompt: str, model_id: str, context_chunks: list) -> dict:
    """
    Calls Amazon Bedrock as fallback.
    """
    messages = []
    if context_chunks:
        context = "\n\n".join(context_chunks)
        messages.append({
            "role": "user",
            "content": f"Context:\n\n{context}\n\nQuestion: {prompt}"
        })
    else:
        messages.append({"role": "user", "content": prompt})

    start = time.time()

    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": messages
        }),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())
    latency_ms = int((time.time() - start) * 1000)
    usage = result.get("usage", {})

    return {
        "answer": result.get("content", [{}])[0].get("text", ""),
        "model": model_id,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "latency_ms": latency_ms,
        "source": "bedrock_fallback"
    }


def emit_metrics(result: dict, used_fallback: bool, timed_out: bool):
    """Publishes inference metrics to CloudWatch."""
    try:
        dimensions = [{"Name": "Environment", "Value": ENVIRONMENT}]
        metrics = []

        if used_fallback:
            metrics.append({
                "MetricName": "BedrockFallbackCount",
                "Dimensions": dimensions,
                "Value": 1,
                "Unit": "Count"
            })
            metrics.append({
                "MetricName": "BedrockFallbackLatency",
                "Dimensions": dimensions,
                "Value": result.get("latency_ms", 0),
                "Unit": "Milliseconds"
            })
        else:
            metrics.append({
                "MetricName": "LocalInferenceCount",
                "Dimensions": dimensions,
                "Value": 1,
                "Unit": "Count"
            })
            metrics.append({
                "MetricName": "LocalInferenceLatency",
                "Dimensions": dimensions,
                "Value": result.get("latency_ms", 0),
                "Unit": "Milliseconds"
            })

        if timed_out:
            metrics.append({
                "MetricName": "LocalTimeoutCount",
                "Dimensions": dimensions,
                "Value": 1,
                "Unit": "Count"
            })

        cloudwatch.put_metric_data(
            Namespace="aws-hybrid-llm",
            MetricData=metrics
        )
    except Exception as e:
        logger.warning(f"Metrics emit failed: {e}")


def lambda_handler(event, context):
    """
    Input:
      {
        "action": "infer" | "warmup",
        "query": str,
        "context_chunks": list (optional),
        "model_override": str (optional)
      }

    Output:
      {
        "answer": str,
        "model": str,
        "source": "local" | "bedrock_fallback",
        "input_tokens": int,
        "output_tokens": int,
        "latency_ms": int,
        "fallback_triggered": bool
      }
    """
    action = event.get("action", "infer")
    config = get_config()

    # ── Warmup ping ──────────────────────────────────────────────
    if action == "warmup":
        try:
            req = urllib.request.Request(
                f"{OLLAMA_ENDPOINT}/api/tags",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
            logger.info("Warmup ping successful")
            return {"status": "warm"}
        except Exception as e:
            logger.warning(f"Warmup ping failed (may be cold): {e}")
            return {"status": "cold"}

    # ── Inference ────────────────────────────────────────────────
    query = event.get("query", "")
    context_chunks = event.get("context_chunks", [])
    model = event.get("model_override") or config["default_model"]

    if not query:
        return {"statusCode": 400, "error": "query is required"}

    used_fallback = False
    timed_out = False
    result = {}

    # Check ECS is running (scale up if needed)
    ecs_ready = ensure_ecs_running()

    # Attempt local Ollama inference
    if ecs_ready and OLLAMA_ENDPOINT:
        try:
            logger.info(f"Calling local Ollama model={model}")
            result = call_ollama(
                prompt=query,
                model=model,
                context_chunks=context_chunks,
                timeout_ms=config["local_timeout_ms"]
            )
            logger.info(f"Local inference success — {result['latency_ms']}ms, "
                        f"{result['output_tokens']} output tokens")

        except TimeoutError:
            timed_out = True
            logger.warning(f"Local LLM timeout after {config['local_timeout_ms']}ms")

        except Exception as e:
            logger.warning(f"Local LLM error: {e}")
            timed_out = True

    else:
        logger.info("ECS not ready (cold start) — using Bedrock fallback this request")
        timed_out = True

    # Bedrock fallback if local failed/timed out
    if timed_out or not result:
        if config["fallback_enabled"]:
            used_fallback = True
            logger.info(f"Falling back to Bedrock model={config['bedrock_fallback_model']}")
            result = call_bedrock(
                prompt=query,
                model_id=config["bedrock_fallback_model"],
                context_chunks=context_chunks
            )
            result["fallback_triggered"] = True
        else:
            return {
                "statusCode": 503,
                "error": "Local LLM unavailable and Bedrock fallback disabled"
            }

    emit_metrics(result, used_fallback, timed_out)

    return {
        **result,
        "fallback_triggered": used_fallback
    }
