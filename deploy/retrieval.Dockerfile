FROM python:3.12-slim

ARG TORCH_VERSION=2.11.0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app/deploy/src \
    HF_HOME=/cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"
COPY deploy/requirements-retrieval.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN useradd --create-home --uid 10001 scentai \
    && mkdir -p /cache/huggingface \
    && chown -R scentai:scentai /cache
COPY src/scentai/__init__.py src/scentai/retrieval.py /app/src/scentai/
COPY deploy/src /app/deploy/src
USER scentai

EXPOSE 8020
CMD ["uvicorn", "scentai_deploy.retrieval_api:app", "--host", "0.0.0.0", "--port", "8020", "--workers", "1"]
