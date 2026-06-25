# docker/ollama/Dockerfile
#
# Builds an Ollama image with the model pre-pulled.
# Push this to ECR instead of using ollama/ollama:latest
# to avoid model download at container startup (air-gap safe).
#
# Usage:
#   docker build --build-arg MODEL=llama3.2 -t ollama-llama3 .
#   docker tag ollama-llama3 <account>.dkr.ecr.<region>.amazonaws.com/aws-hybrid-llm/ollama:llama3.2
#   docker push <account>.dkr.ecr.<region>.amazonaws.com/aws-hybrid-llm/ollama:llama3.2

FROM ollama/ollama:latest

ARG MODEL=llama3.2

# Pre-pull model during image build
# Model weights baked into image layer — no download on startup
RUN ollama serve & \
    sleep 5 && \
    ollama pull ${MODEL} && \
    pkill ollama

# Expose Ollama API port
EXPOSE 11434

# Start Ollama server
ENTRYPOINT ["ollama", "serve"]
