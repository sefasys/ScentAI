FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app/deploy/src

WORKDIR /app
COPY deploy/requirements-api.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN useradd --create-home --uid 10001 scentai
COPY src/scentai/__init__.py src/scentai/orchestrator.py /app/src/scentai/
COPY deploy/src /app/deploy/src
USER scentai

EXPOSE 8080
CMD ["uvicorn", "scentai_deploy.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
