FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
COPY orca/ ./orca/

RUN uv pip install --system -e .

RUN mkdir -p /root/.orca

EXPOSE 7337

ENV ORCA_HOME=/root/.orca \
    ORCA_OLLAMA_HOST=http://ollama:11434

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:7337/api/status || exit 1

CMD ["orca", "serve", "--host", "0.0.0.0", "--port", "7337", "--no-open"]
