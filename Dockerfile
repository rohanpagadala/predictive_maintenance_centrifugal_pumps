FROM python:3.10-slim AS base

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pdm_utils.py model_persistence.py model_registry.py ./
COPY app/ ./app/
COPY dashboard/ ./dashboard/
COPY models/ ./models/
COPY outputs/ ./outputs/

RUN mkdir -p logs

ENV PYTHONUNBUFFERED=1 \
    PDM_MODELS_DIR=/srv/app/models \
    PDM_LOGS_DIR=/srv/app/logs \
    PDM_OUTPUTS_DIR=/srv/app/outputs

EXPOSE 8000 8501

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
