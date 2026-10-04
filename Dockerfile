FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY support_agent support_agent
COPY config config
COPY knowledge knowledge
COPY collections collections
ENV AGENT_DATA_DIR=/data/agent
EXPOSE 8001
CMD ["sh", "-c", "uvicorn support_agent.server:app --host 0.0.0.0 --port ${PORT:-8001}"]
