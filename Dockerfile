# Single image, two services. docker-compose.yml runs this same image twice
# with different `command:` overrides -- one for the FastAPI backend, one
# for the Streamlit dashboard -- so there is exactly one build to maintain.
FROM python:3.10-slim AS base

# libgomp1 is required by LightGBM/XGBoost's OpenMP-based training/inference.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and already-trained model artifacts only -- no raw
# dataset, notebook, or tests (see .dockerignore). Models are NOT retrained
# here; they're the same joblib files produced by the training notebook.
COPY pdm_utils.py model_persistence.py ./
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

# Default command runs the API; docker-compose.yml's `dashboard` service
# overrides this with the streamlit command.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
