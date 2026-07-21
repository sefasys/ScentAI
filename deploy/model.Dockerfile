FROM vllm/vllm-openai:v0.25.1-x86_64-cu129

COPY deploy/model_server/entrypoint.py /opt/scentai/entrypoint.py
ENTRYPOINT ["python3", "/opt/scentai/entrypoint.py"]
