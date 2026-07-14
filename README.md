# Predictive Maintenance (PdM) — Industrial Centrifugal Pumps

End-to-end ML system: EDA → feature engineering → model training/comparison →
persistence → **FastAPI backend + Streamlit dashboard, Dockerized and ready
to deploy**. Predicts (1) failure severity classification and (2) remaining
useful life (RUL) regression from pump telemetry.

```json
{
  "failure_state": "Warning",
  "confidence": 0.9998,
  "remaining_useful_life": 120.4
}
```

---

## 1. Project Overview

Unplanned failures of industrial centrifugal pumps cause costly downtime.
This project turns hourly pump telemetry (vibration, temperature, pressure,
flow rate) into two decision-support signals for plant operators:

1. **Failure classification** — Normal / Warning / Critical /
   Bearing_Failure / Motor_Failure / Seal_Failure (6-class).
2. **Remaining Useful Life (RUL) regression** — hours until failure.

The project has two halves, built in two passes:

- **ML pipeline** (`Predictive_Maintenance_Pipeline.ipynb` + `pdm_utils.py`):
  dataset understanding, EDA, preprocessing, feature engineering, feature
  selection, training 4 classifiers × 4 regressors, evaluation, comparison,
  and persistence. Fully documented inline in the notebook.
- **Serving layer** (`app/`, `dashboard/`, `model_persistence.py`, `Dockerfile`):
  loads the already-trained models (**no retraining**) and exposes them via
  a FastAPI backend and a Streamlit dashboard, containerized for deployment.

---

## 2. Folder Structure

```
AI_Predictive/
│
├── app/                          # FastAPI backend + inference pipeline
│   ├── api.py                    # FastAPI app: routes, middleware, error handling
│   ├── predict.py                # load_artifacts / preprocess_input / engineer_features /
│   │                              #   predict_failure / predict_rul / predict / predict_batch
│   ├── preprocessing.py          # raw input -> pdm_utils-shaped DataFrame, imputation, safe encoding
│   ├── feature_engineering.py    # applies pdm_utils feature engineering via the saved config
│   ├── model_loader.py           # loads + caches model_persistence artifacts once per process
│   ├── monitoring.py             # in-process request/latency/prediction counters (GET /metrics)
│   ├── schemas.py                # Pydantic request/response models
│   ├── utils.py                  # logging setup, timing helper, custom exceptions
│   └── config.py                 # paths, env-var driven settings
│
├── dashboard/
│   └── streamlit_app.py          # 8-page dashboard (calls the API for predictions)
│
├── models/                       # joblib artifacts (Phase 1 persistence) -- NOT retrained by the app
│   ├── classification_model_lightgbm.joblib
│   ├── regression_model_xgboost.joblib
│   ├── feature_scaler.joblib
│   ├── label_encoders.joblib
│   ├── selected_features.joblib
│   ├── class_names.joblib
│   ├── feature_engineering_config.joblib
│   └── metadata.json
│
├── data/                         # raw training dataset (not needed at serving time)
├── outputs/figures/              # plots exported by the training notebook
├── logs/                         # api.log, predictions.log, errors.log (rotating)
├── tests/                        # pytest suite (31 tests)
│
├── pdm_utils.py                  # training-pipeline functions (shared with app/ for identical preprocessing)
├── model_persistence.py          # save_models() / load_models() / predict()
├── Predictive_Maintenance_Pipeline.ipynb
│
├── Dockerfile                    # one image, two services (api + dashboard via different commands)
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── README.md
└── .env
```

---

## 3. Installation

```bash
git clone <this repo>
cd AI_Predictive
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the two services locally (two terminals):

```bash
# Terminal 1 — API
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs

# Terminal 2 — Dashboard
PDM_API_URL=http://localhost:8000 streamlit run dashboard/streamlit_app.py
# http://localhost:8501
```

Or run both with Docker — see [§10](#10-docker).

---

## 4. Dataset Description

10,000 hourly readings from 20 pumps (`AI_Predictive_Maintenance_Pump_Dataset_10000.xlsx`):

| Column | Description |
|---|---|
| `Timestamp`, `Asset_ID` | reading time; pump identifier |
| `Machine_Model`, `Location` | pump design; site (Location found to behave as noise, not used as a feature) |
| `Vibration_mm_s`, `Temperature_C`, `Pressure_psi`, `Flow_Rate_m3_h` | telemetry |
| `Failure_State` | Normal (9000) / Warning (800) / Critical (180) / Bearing_Failure (9) / Motor_Failure (7) / Seal_Failure (4) |
| `RUL_Hours` | hours remaining until failure (0 at a failure event) |

**Data-reality findings that shaped the pipeline** (from actually profiling
the file, not just its spec): 20 pumps are interleaved on one hourly log, so
each pump is sampled once every 20 hours; each pump runs repeated ~500-hour
degradation cycles; and raw sensor bands are non-overlapping for
Normal/Warning/Critical but overlap heavily across the three failure-type
classes, which total only 20 rows from a single pump. Full detail is in the
notebook's Stage 1–2.

---

## 5. ML Pipeline

All 9 stages run in `Predictive_Maintenance_Pipeline.ipynb`, executed
end-to-end with zero errors:

1. Dataset Understanding — feature purposes, business problem, ML approach.
2. EDA — distributions, class balance, correlation heatmap, outliers, insights after every plot.
3. Preprocessing — chronological per-asset sort, dedup, per-asset interpolation, label encoding, 6-class target construction.
4. Feature Engineering — see [§6](#6-feature-engineering).
5. Feature Selection — correlation pruning (53→42) + RF importance + Mutual Information + RFE → 34 final features.
6. Data Preparation — **group-aware chronological 80:20 split** (not random — a random split would leak near-identical adjacent readings across train/test).
7. Model Training — Decision Tree, Random Forest, XGBoost, LightGBM for both tasks.
8. Model Comparison — ranked tables, per-class recall breakdown, deployment recommendation.
9. Model Persistence — save/load/round-trip-verify the selected models (this is what `models/` contains).

Run it: `jupyter nbconvert --to notebook --execute --inplace Predictive_Maintenance_Pipeline.ipynb`

---

## 6. Feature Engineering

Computed per-`Asset_ID` (never mixing one pump's history with another's):

- **Rolling mean/std** over the last 4 / 12 / 24 *readings* (reinterpreted from
  "hours" since each pump is only sampled every 20 hours — a calendar-hour
  window would be empty).
- **Lag features** `t-1, t-2, t-3`.
- **Rate of change & percent change** per telemetry channel.
- **Interaction terms**: Temperature×Pressure, Vibration×Temperature, Vibration×Pressure, Flow×Pressure.

At inference time, `app/feature_engineering.py` reproduces this **exactly**
by calling the same `pdm_utils` functions with the window/lag sizes read
from the persisted `feature_engineering_config.joblib` — not a
reimplementation, so "identical to training" is true by construction.

---

## 7. Models Used & Results

| Task | Algorithms compared | Selected | Why |
|---|---|---|---|
| Classification (`Failure_Class`, 6 classes) | Decision Tree, Random Forest, XGBoost, **LightGBM** | **LightGBM** | Best macro Recall (0.80) — chosen programmatically from Stage 8's comparison table, not hardcoded |
| Regression (`RUL_Hours`) | Decision Tree, Random Forest, **XGBoost**, LightGBM | **XGBoost** | Best cross-validated MAE (36.8h) |

**Honest result, stated plainly:** overall classification accuracy is
~0.999, but that's misleading alone — Normal/Warning/Critical are caught
near-perfectly (non-overlapping sensor bands in the data), while the three
failure-type classes (Bearing/Motor/Seal_Failure) have only 4–9 training
examples total, all from one pump, and their recall is correspondingly
unstable (0.0–1.0 depending on the model and which cycles landed in test).
**This is a data-scarcity ceiling, not a fixable modeling problem** — see
the notebook's Stage 7–8 for the full analysis, and the dashboard's *About*
page for the same caveat surfaced to end users.

Regression: MAE ≈ 37–40 hours, R² ≈ 0.86–0.88 across all 4 models.

---

## 8. API Usage

Interactive docs at `/docs` (Swagger) once running.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Liveness check |
| GET | `/health` | Readiness check + model identity |
| GET | `/metrics` | Request count, failure rate, latency percentiles, model load time |
| POST | `/predict` | Single pump reading (or history) → JSON prediction |
| POST | `/predict-csv` | CSV upload → downloadable CSV of predictions |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"readings":[{"machine_model":"Nova-P","vibration_mm_s":4.5,
       "temperature_c":80.0,"pressure_psi":135.0,"flow_rate_m3_h":235.0}]}'

curl -X POST http://localhost:8000/predict-csv \
  -F "file=@pumps.csv" -o predictions.csv
```

A submitted `readings` list may be a single reading (rolling/lag features
then degrade gracefully to that one point) or a chronological history for
one pump (recommended: last ~24 readings) for materially better feature
quality. Unknown `machine_model` values, missing telemetry channels (with
history to interpolate from), and out-of-order/missing timestamps are all
handled — see `app/preprocessing.py` docstrings for exact behavior.

---

## 9. Streamlit Usage

```bash
PDM_API_URL=http://localhost:8000 streamlit run dashboard/streamlit_app.py
```

8 pages: **Home** (API status at a glance) · **Project Overview** ·
**Single Prediction** (manual form → live prediction + probability chart) ·
**Batch Prediction** (CSV upload → results table, charts, CSV download) ·
**Model Performance** (metrics + notebook-exported plots) ·
**Feature Importance** (live bar charts from the loaded models) ·
**Prediction History** (session log + persisted `logs/predictions.log`) ·
**About**.

The dashboard calls the FastAPI backend over HTTP for predictions (same
inference code path as `curl`/any other client) and reads `models/` /
`outputs/figures/` directly for read-only introspection pages.

---

## 10. Docker

One `Dockerfile`, two `docker-compose.yml` services (`api`, `dashboard`)
built from the same image, sharing the `models/` folder as a read-only
volume:

```bash
docker build -t pdm-app:latest .
docker compose up -d          # starts both containers
docker compose logs -f api    # tail logs
docker compose down           # stop and remove containers
```

- API → `http://localhost:8000` (health-checked via `/health`)
- Dashboard → `http://localhost:8501` (health-checked via `/_stcore/health`,
  waits for the API's healthcheck via `depends_on: condition: service_healthy`)

> **Note:** the Dockerfile and compose file were written and syntax-validated
> (`docker compose config`) in this environment, but an actual `docker build`
> could not be executed here (no Docker daemon available in this sandbox) —
> verify the build on a machine with Docker running before deploying.

---

## 11. Deployment Guide

General prerequisites for any target below: `models/` must be present in
the deployed environment (either baked into the image at build time, as the
`Dockerfile` does, or mounted/uploaded separately) — the app never trains,
it only loads.

### Render

1. Push this repo to GitHub.
2. New **Web Service** → connect the repo → Environment: **Docker**.
3. Two services (or one + a background worker) since Render's free tier
   runs one process per service:
   - `pdm-api`: Dockerfile build, expose port `8000`, health check path `/health`.
   - `pdm-dashboard`: same Dockerfile, override start command to the
     `streamlit run ...` command from `docker-compose.yml`, set env var
     `PDM_API_URL=https://pdm-api.onrender.com`.
4. Env vars: `PDM_LOG_LEVEL=INFO` (Render provides `PORT` automatically —
   update `CMD`/start command to bind `--port $PORT` instead of the fixed
   8000/8501 if using Render's dynamic port).
5. Render provisions HTTPS automatically on `*.onrender.com`.

### Azure App Service (Web App for Containers)

```bash
az group create -n pdm-rg -l eastus
az acr create -n pdmregistry -g pdm-rg --sku Basic
az acr build --registry pdmregistry --image pdm-app:latest .

az appservice plan create -n pdm-plan -g pdm-rg --is-linux --sku B1
az webapp create -n pdm-api -g pdm-rg --plan pdm-plan \
  --deployment-container-image-name pdmregistry.azurecr.io/pdm-app:latest
az webapp config appsettings set -n pdm-api -g pdm-rg \
  --settings PDM_HOST=0.0.0.0 WEBSITES_PORT=8000
az webapp config container set -n pdm-api -g pdm-rg \
  --container-command-line "uvicorn app.api:app --host 0.0.0.0 --port 8000"
# Repeat for pdm-dashboard with the streamlit command and WEBSITES_PORT=8501,
# PDM_API_URL=https://pdm-api.azurewebsites.net
```

App Service terminates TLS at the platform edge automatically
(`*.azurewebsites.net` is HTTPS by default; add a custom domain + managed
certificate for a branded URL). Scale via the App Service Plan tier /
autoscale rules on CPU%.

### AWS EC2

```bash
# On a fresh EC2 instance (Amazon Linux 2023 / Ubuntu), with Docker installed:
git clone <repo> && cd AI_Predictive
docker build -t pdm-app:latest .
docker compose up -d
```

- Open inbound Security Group rules for 8000/8501 (or better, only 80/443 —
  see HTTPS note below).
- **HTTPS**: EC2 has no built-in TLS termination. Put an
  **Application Load Balancer** (ACM-issued certificate) or an **nginx +
  certbot** reverse proxy in front of the two containers instead of exposing
  8000/8501 directly.
- **Scaling**: single EC2 instance is fine for light load; for real scale
  put the containers in an **ECS/Fargate service** behind the same ALB, with
  a target-tracking autoscaling policy on CPU or request count.

### Google Cloud Run

Cloud Run is a strong fit here — it's built for exactly this
"stateless container behind HTTPS, scales to zero" shape, and needs the API
and dashboard split into **two separate Cloud Run services** (Cloud Run runs
one process per container):

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/pdm-app

gcloud run deploy pdm-api \
  --image gcr.io/PROJECT_ID/pdm-app --port 8000 \
  --set-env-vars PDM_HOST=0.0.0.0 \
  --command uvicorn --args app.api:app,--host,0.0.0.0,--port,8000

gcloud run deploy pdm-dashboard \
  --image gcr.io/PROJECT_ID/pdm-app --port 8501 \
  --set-env-vars PDM_API_URL=https://pdm-api-xxxxx.a.run.app \
  --command streamlit --args run,dashboard/streamlit_app.py,--server.port,8501,--server.address,0.0.0.0,--server.headless,true
```

HTTPS is automatic on `*.run.app`. Scaling is automatic and to-zero by
default (set `--min-instances 1` on `pdm-api` if cold-start latency after
idle periods is a problem, since model loading takes ~150ms but the
container cold-start itself is slower).

### Production best practices (all targets)

- Lock down CORS (`app/api.py`'s `CORSMiddleware` currently allows `*` for
  local dev convenience — restrict `allow_origins` to the dashboard's real
  origin in production).
- Set `PDM_LOG_LEVEL=WARNING` in production to cut log volume; keep
  `logs/` on a persistent volume or ship it to a log aggregator.
- Run the API with multiple workers behind a process manager
  (`uvicorn ... --workers 4` or gunicorn+uvicorn workers) once traffic
  exceeds a single worker's capacity — model artifacts are loaded once per
  worker process (not shared across workers), so pick worker count based on
  available memory (~3MB per loaded model set is small, this is not a
  binding constraint here).
- Monitor `GET /metrics` (failure rate, p95 latency) with whatever
  monitoring stack the platform provides; alert on `failure_rate` climbing
  or `model_load_time_ms` being `null` (means artifacts never loaded).

---

## 12. Logging & Monitoring

Three rotating log files in `logs/`: `api.log` (all requests + info),
`errors.log` (warnings/errors only), `predictions.log` (one line per
prediction, single or batch-count). `GET /metrics` exposes live counters:
total/failed requests, failure rate, prediction count, average & p95
latency, and model load time — see `app/monitoring.py`.

---

## 13. Testing

```bash
python3 -m pytest -v
```

31 tests across 4 files — `tests/test_model_persistence.py` (save/load round
trip fidelity, error handling, with throwaway dummy models so the real
`models/` artifacts are never touched), `tests/test_model_loader.py`
(caching), `tests/test_predict.py` (single + batch prediction, missing
values, unknown categories), `tests/test_api.py` (all 5 endpoints, status
codes, error mapping). All tests run against the **real trained models** —
these are integration tests, not mocks, since the whole point is confirming
the persisted artifacts actually work.

---

## 14. Screenshots

*(placeholder — add screenshots of the Streamlit dashboard's Single
Prediction, Batch Prediction, and Model Performance pages here once
deployed)*

---

## 15. Future Improvements

- Collect more failure-type examples across multiple pumps before trusting
  Bearing/Motor/Seal_Failure predictions in production (currently 4–9
  samples total, one asset — see [§7](#7-models-used--results)).
- Add authentication/rate-limiting to the API before exposing it publicly.
- Move `Prediction History` from log-file parsing to a real datastore
  (Postgres/SQLite) for queryable, multi-instance-safe history.
- Add a `/predict-stream` websocket endpoint for continuous live telemetry
  rather than request/response batches.
- Wire `GET /metrics` into Prometheus/Grafana instead of the current simple
  JSON snapshot, once traffic justifies it.
