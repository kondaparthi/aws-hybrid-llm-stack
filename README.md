# 🖥️ aws-hybrid-llm-stack
### Local LLM on ECS Fargate + Bedrock Fallback — CloudFormation Deployable

[![AWS](https://img.shields.io/badge/AWS-CloudFormation-orange?logo=amazonaws)](https://aws.amazon.com/cloudformation/)
[![ECS](https://img.shields.io/badge/AWS-ECS%20Fargate-blue?logo=amazonaws)](https://aws.amazon.com/fargate/)
[![GovCloud](https://img.shields.io/badge/AWS-GovCloud%20Ready-blue)](https://aws.amazon.com/govcloud-us/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

> **Run open-source LLMs (Llama, Mistral, Phi, Gemma) inside your AWS VPC on ECS Fargate.**
> Automatic Bedrock fallback when local LLM is unavailable or times out.
> Zero GPU required. Scale to zero when idle. Deploy in GovCloud.

---

## Architecture

```
aws-ai-router
     │
     │  POST /api/chat
     ▼
Inference Proxy Lambda (VPC)
     │
     ├── ECS Running? ──YES──► Ollama on Fargate ──► Answer
     │                                   ▲
     │                         (llama3.2 / mistral / phi3)
     │
     └── ECS Cold/Timeout ──► Bedrock Fallback ──► Answer
                              (Claude Haiku / Nova)
```

---

## Supported Models

| Model | Min Memory | Fargate Size | Quality |
|---|---|---|---|
| `llama3.2` | 16 GB | 4vCPU / 16GB | ⭐⭐⭐⭐ |
| `llama3.2:1b` | 8 GB | 2vCPU / 8GB | ⭐⭐⭐ |
| `mistral` | 16 GB | 4vCPU / 16GB | ⭐⭐⭐⭐ |
| `phi3` | 8 GB | 2vCPU / 8GB | ⭐⭐⭐ |
| `gemma2:2b` | 8 GB | 2vCPU / 8GB | ⭐⭐⭐ |

---

## Repo Structure

```
aws-hybrid-llm-stack/
├── cloudformation/
│   ├── main.yaml              # Root nested stack
│   ├── networking.yaml        # VPC, subnets, SGs, VPC endpoints
│   ├── ecs-ollama.yaml        # ECS cluster, task def, ALB, autoscaling
│   └── bedrock-fallback.yaml  # Inference proxy Lambda, model manager, SSM
├── lambda/
│   ├── inference_proxy/       # Ollama caller + Bedrock fallback
│   │   └── handler.py
│   └── model_manager/         # Pull/switch models, scale ECS
│       └── handler.py
├── docker/
│   └── ollama/
│       └── Dockerfile         # Pre-baked model image for air-gap deploy
├── tests/
│   └── test_inference_proxy.py
└── README.md
```

---

## Deploy

```bash
# 1. Package and upload
./scripts/package.sh your-artifact-bucket dev us-east-1

# 2. Deploy stack
aws cloudformation deploy \
  --template-file cloudformation/main.yaml \
  --stack-name aws-hybrid-llm-dev \
  --parameter-overrides \
      Environment=dev \
      DefaultModel=llama3.2 \
      DesiredCount=1 \
      AutoShutdownEnabled=true \
      ArtifactBucket=your-artifact-bucket \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --region us-east-1

# 3. Get inference endpoint (pass to aws-ai-router)
aws cloudformation describe-stacks \
  --stack-name aws-hybrid-llm-dev \
  --query "Stacks[0].Outputs[?OutputKey=='InferenceEndpoint'].OutputValue" \
  --output text
```

## Switch Models (no redeploy)

```bash
# Pull a new model to running Ollama
aws lambda invoke \
  --function-name hybrid-llm-dev-model-manager \
  --payload '{"action":"pull_model","model":"mistral"}' \
  response.json

# Switch active model (updates SSM)
aws lambda invoke \
  --function-name hybrid-llm-dev-model-manager \
  --payload '{"action":"switch_model","model":"mistral"}' \
  response.json
```

---

## GovCloud Notes

- All compute stays inside your VPC
- VPC endpoints for Bedrock, ECR, and S3 — no internet egress required
- For full air-gap: build model into Docker image (see `docker/ollama/Dockerfile`)
- Compatible with `us-gov-west-1` and `us-gov-east-1`

---

## Part of aws-ai-platform

This project is the **compute layer** consumed by [`aws-ai-router`](../aws-ai-router).
Pass the `InferenceEndpoint` output to the router's `LocalLLMEndpoint` parameter.

> Local LLM (Route A + B) → this stack
> Bedrock (Route C) → direct from router
> RAG retrieval → [`aws-rag-pipeline`](../aws-rag-pipeline)
