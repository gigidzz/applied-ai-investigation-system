# Container image for the investigation service.
#
# The embedding model runs under ONNX Runtime rather than PyTorch: the same
# weights producing the same vectors, measured at 0.999999 cosine similarity,
# in 345 MB of memory instead of 626 MB. PyTorch's training machinery was
# 489 MB of the original footprint and none of it was used, and its Linux
# wheel additionally bundles CUDA for a GPU no free tier provides.
#
# The model is downloaded at build time rather than on the first request,
# because a cold container answering its first question should not also be
# fetching weights while someone waits.

FROM python:3.12-slim

# Most hosts run the container as a non-root user. Anything the process writes
# must be owned by it: the model cache and the uploads directory.
RUN useradd -m -u 1000 app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/home/app/.cache/fastembed \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first, so editing source does not reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
COPY data/processed/sparse_stats_structural.json ./data/processed/
RUN uv sync --frozen --no-dev

RUN mkdir -p /home/app/.cache /app/data/corpus/uploads && chown -R app:app /home/app /app
USER app

# Bake the model into the image, as the runtime user so the cache is readable.
RUN python -c "from fastembed import TextEmbedding; \
    list(TextEmbedding(model_name='BAAI/bge-small-en-v1.5').embed(['warm up']))"

# Cloud Run injects the port as $PORT and marks the revision unhealthy if the
# process binds anything else. Other hosts expect a fixed port, so the
# variable is honoured with a default. The shell form is required: the exec
# form does not expand it.
EXPOSE 8080
CMD ["sh", "-c", "uvicorn ragsys.api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
