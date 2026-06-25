"""
model_manager/handler.py

Manages local LLM model lifecycle:
  - List available models on running Ollama instance
  - Pull new model to running instance
  - Switch active model (update SSM)
  - ECS scale up/down on demand
  - Report model status and memory usage
"""

import os
import json
import time
import logging
import boto3
import urllib.request

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ecs = boto3.client("ecs")
ssm = boto3.client("ssm")

ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")
OLLAMA_ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "")
ECS_CLUSTER = os.environ.get("ECS_CLUSTER", "")
ECS_SERVICE = os.environ.get("ECS_SERVICE", "")
SSM_PREFIX = f"/aws-hybrid-llm/{ENVIRONMENT}"

SUPPORTED_MODELS = {
    "llama3.2":    {"min_memory_gb": 8,  "recommended_memory_gb": 16},
    "llama3.2:1b": {"min_memory_gb": 4,  "recommended_memory_gb": 8},
    "mistral":     {"min_memory_gb": 8,  "recommended_memory_gb": 16},
    "phi3":        {"min_memory_gb": 4,  "recommended_memory_gb": 8},
    "gemma2:2b":   {"min_memory_gb": 4,  "recommended_memory_gb": 8},
}


def ollama_request(path: str, method: str = "GET", data: dict = None) -> dict:
    url = f"{OLLAMA_ENDPOINT}{path}"
    payload = json.dumps(data).encode() if data else None
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"} if payload else {},
        method=method
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def list_models() -> dict:
    """List all models currently loaded on Ollama."""
    try:
        result = ollama_request("/api/tags")
        models = result.get("models", [])
        return {
            "status": "ok",
            "models": [
                {
                    "name": m["name"],
                    "size_gb": round(m.get("size", 0) / 1e9, 2),
                    "modified": m.get("modified_at", "")
                }
                for m in models
            ],
            "count": len(models)
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "models": []}


def pull_model(model_name: str) -> dict:
    """Pull a model to the running Ollama instance."""
    if model_name not in SUPPORTED_MODELS:
        return {
            "status": "error",
            "error": f"Model {model_name} not in supported list",
            "supported": list(SUPPORTED_MODELS.keys())
        }

    logger.info(f"Pulling model: {model_name}")
    try:
        # Ollama pull is streaming — we send and wait
        payload = json.dumps({"name": model_name, "stream": False}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_ENDPOINT}/api/pull",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=280) as resp:
            result = json.loads(resp.read())

        logger.info(f"Model pull complete: {model_name}")
        return {"status": "ok", "model": model_name, "result": result}

    except Exception as e:
        logger.error(f"Model pull failed: {e}")
        return {"status": "error", "model": model_name, "error": str(e)}


def switch_model(model_name: str) -> dict:
    """Update SSM to change the active default model."""
    if model_name not in SUPPORTED_MODELS:
        return {
            "status": "error",
            "error": f"Unsupported model: {model_name}"
        }

    # Verify model is pulled
    models_result = list_models()
    pulled = [m["name"] for m in models_result.get("models", [])]
    if not any(model_name in p for p in pulled):
        return {
            "status": "error",
            "error": f"Model {model_name} not pulled yet. Call pull_model first."
        }

    # Update SSM
    ssm.put_parameter(
        Name=f"{SSM_PREFIX}/default-model",
        Value=model_name,
        Type="String",
        Overwrite=True
    )
    logger.info(f"Active model switched to: {model_name}")
    return {
        "status": "ok",
        "active_model": model_name,
        "note": "SSM updated. New requests will use this model."
    }


def scale_service(desired_count: int) -> dict:
    """Scale ECS Ollama service up or down."""
    if desired_count < 0 or desired_count > 5:
        return {"status": "error", "error": "desired_count must be 0–5"}

    try:
        ecs.update_service(
            cluster=ECS_CLUSTER,
            service=ECS_SERVICE,
            desiredCount=desired_count
        )
        logger.info(f"ECS service scaled to {desired_count}")
        return {
            "status": "ok",
            "desired_count": desired_count,
            "note": f"Service scaling to {desired_count} tasks"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_service_status() -> dict:
    """Get current ECS service health and task count."""
    try:
        response = ecs.describe_services(
            cluster=ECS_CLUSTER,
            services=[ECS_SERVICE]
        )
        svc = response["services"][0]
        return {
            "status": "ok",
            "desired_count": svc["desiredCount"],
            "running_count": svc["runningCount"],
            "pending_count": svc["pendingCount"],
            "service_status": svc["status"],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def lambda_handler(event, context):
    """
    Actions:
      list_models    — list models on Ollama
      pull_model     — pull a model { model: str }
      switch_model   — change active model { model: str }
      scale          — scale ECS { desired_count: int }
      status         — ECS service status
    """
    action = event.get("action", "status")

    if action == "list_models":
        return list_models()

    elif action == "pull_model":
        model = event.get("model")
        if not model:
            return {"status": "error", "error": "model is required"}
        return pull_model(model)

    elif action == "switch_model":
        model = event.get("model")
        if not model:
            return {"status": "error", "error": "model is required"}
        return switch_model(model)

    elif action == "scale":
        count = event.get("desired_count")
        if count is None:
            return {"status": "error", "error": "desired_count is required"}
        return scale_service(int(count))

    elif action == "status":
        return {
            **get_service_status(),
            "models": list_models().get("models", []),
            "ollama_endpoint": OLLAMA_ENDPOINT,
            "environment": ENVIRONMENT
        }

    else:
        return {
            "status": "error",
            "error": f"Unknown action: {action}",
            "valid_actions": ["list_models", "pull_model", "switch_model", "scale", "status"]
        }
