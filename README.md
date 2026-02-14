# Open-Source Fine-Tune Configuration Engine

Deterministic recommendation engine for fine-tuning setups with:

- V1 dataset normalization and deduplication
- Statistical profile extraction by task/model-size/adapter
- GPU/platform-aware safety constraints
- Dependency pinning by platform
- PostgreSQL/SQLite persistence for configs, profiles, and recommendation history
- Template-based notebook generation
- Multi-strategy recommendation with auto routing (`auto`, `deterministic`, `hybrid`, `hybrid_ml`)

## Quick Start

### Local venv (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --clear .venv
. .venv/bin/activate
uv pip install -r requirements.txt
```

```bash
PYTHONPATH=src python3 scripts/build_profiles.py \
  --dataset finetuning_configs_final.jsonl
```

```bash
PYTHONPATH=src python3 scripts/train_ml_reranker.py \
  --dataset finetuning_configs_final.jsonl \
  --out-model artifacts/ml_reranker.joblib \
  --out-metrics artifacts/ml_reranker_metrics.json
```

```bash
PYTHONPATH=src python3 scripts/train_hp_predictor.py \
  --dataset finetuning_configs_final.jsonl \
  --out-model artifacts/hp_predictor.joblib \
  --out-metrics artifacts/hp_predictor_metrics.json
```

```bash
PYTHONPATH=src python3 scripts/generate_recommendation.py \
  --dataset finetuning_configs_final.jsonl \
  --ml-reranker artifacts/ml_reranker.joblib \
  --hp-predictor artifacts/hp_predictor.joblib \
  --platform colab \
  --plan free \
  --task instruction_following \
  --adapter qlora \
  --strategy auto \
  --rerank-top-k 5 \
  --model-size medium \
  --out-json artifacts/recommendation.json \
  --out-notebook artifacts/recommended_notebook.ipynb
```

```bash
PYTHONPATH=src python3 scripts/validate_notebooks.py \
  --dataset finetuning_configs_final.jsonl \
  --out-dir artifacts/notebook_smoke
```

```bash
PYTHONPATH=src python3 scripts/evaluate_candidate.py \
  --candidate-version 20260212_01 \
  --dataset finetuning_configs_final.jsonl \
  --auto-promote-to-staging
```

```bash
PYTHONPATH=src python3 scripts/run_release_cycles.py \
  --dataset finetuning_configs_final.jsonl \
  --golden evaluation/golden_dataset.jsonl \
  --cycles 3 \
  --prefix canary_cycle
```

## API (FastAPI)

```bash
PYTHONPATH=src uvicorn ft_config_engine.api:app --host 0.0.0.0 --port 8000
```

Environment variables:

- `DATABASE_URL`:
  - PostgreSQL example: `postgresql://ft_user:ft_password@localhost:5432/ft_config_engine`
  - Default fallback: local SQLite in `artifacts/ft_config_engine.db`
- `FT_CONFIG_DATASET_PATH`: defaults to `finetuning_configs_final.jsonl`; supports comma-separated files and auto-discovers `real_world*.jsonl` and `fixed*.jsonl` alongside the primary dataset.
  - Accepts both canonical rows (`task.task_type`, `model.parameter_count`, `training_config.max_seq_length`) and real-world rows (`task_type`, `model.size_params`, `training_config.batch_size`).
- `FT_CONFIG_ML_RERANKER_PATH`: optional path to trained reranker artifact (`artifacts/ml_reranker.joblib` by default). If missing, `hybrid_ml` automatically falls back to `hybrid`.
- `FT_CONFIG_HP_PREDICTOR_PATH`: optional path to trained direct hyperparameter predictor artifact (`artifacts/hp_predictor.joblib` by default). If missing, engine uses statistical baselines only.
- Generated notebook templates support `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` / `HUGGINGFACEHUB_API_TOKEN` for gated model access.
- `strategy` request field defaults to `auto`: engine selects `hybrid_ml` when the ML reranker is loaded, otherwise `hybrid`, then falls back to `deterministic` if needed.
- `sequence_length`, `num_gpus`, and `epochs` are optional request fields. If omitted, the engine auto-selects safe values from corpus priors + GPU constraints.
- `CORS_ORIGINS`: comma-separated origins, default `http://localhost:3000`
- `TRUSTED_HOSTS`: comma-separated hosts, default `localhost,127.0.0.1`
- `REQUIRE_API_KEY`: `true|false`, default `false` unless `API_KEY` is set
- `API_KEY`: shared API key for protected endpoints (header: `X-API-Key`)
- `EXPOSE_INTERNALS`: `true|false`, default `false` (hides dataset/db internals from `/health`)
- `RATE_LIMIT_PER_MINUTE`: default limiter for all routes
- Route-specific limiter overrides: `HEALTH_RATE_LIMIT`, `READ_RATE_LIMIT`, `RECENT_RATE_LIMIT`, `RECOMMEND_RATE_LIMIT`, `BOOTSTRAP_RATE_LIMIT`
  - telemetry limiter: `TELEMETRY_RATE_LIMIT`
- Governance/retraining limiter overrides: `GOVERNANCE_RATE_LIMIT`, `RETRAIN_RATE_LIMIT`
- Canary controls:
  - `CANARY_FRACTION` (0.0-1.0, default `0.0`)
  - `FT_CONFIG_ML_RERANKER_CANARY_PATH`
  - `FT_CONFIG_HP_PREDICTOR_CANARY_PATH`
- Retraining controls:
  - `RETRAIN_CHECK_ON_TELEMETRY` (default `true`)
  - `AUTO_RETRAIN_ON_TELEMETRY` (default `false`)

API endpoints:

- `GET /health`
- `GET /ml-reranker`
- `GET /hp-predictor`
- `GET /brain/strategy`
- `POST /feedback`
- `POST /telemetry/run-start`
- `POST /telemetry/run-complete`
- `GET /telemetry/recent`
- `GET /metrics` (Prometheus format when `prometheus_client` is available)
- `GET /governance/events`
- `GET /governance/model`
- `GET /governance/canary/summary`
- `POST /governance/promote`
- `POST /governance/rollback`
- `GET /retraining/status`
- `POST /retraining/check`
- `POST /bootstrap`
- `GET /profiles`
- `GET /configs`
- `GET /recommendations/recent`
- `POST /recommend`

`GET /ml-reranker` returns reranker load status/version and metadata (full metadata only when `EXPOSE_INTERNALS=true`).
`GET /brain/strategy` returns the learned strategy decision for a task/adapter/model-size context.
`POST /feedback` records user feedback (`rating`, `success`, `notes`) for a recommendation event; this is used by the learning brain to guide future `strategy=auto` decisions.
`POST /telemetry/run-start` and `POST /telemetry/run-complete` ingest run-level telemetry (PII-redacted text fields + anomaly-checked numeric fields) for the production data flywheel.
Governance endpoints are backed by a local model-state machine (with optional MLflow mirroring when `MLFLOW_TRACKING_URI` is set).
Retraining endpoints expose trigger diagnostics and optional one-shot retraining execution.
Canary summary endpoint reports traffic split + confidence + risk indicators by model slot.

## Docker

```bash
docker build -t ft-config-engine:latest .
docker run --rm -p 8000:8000 ft-config-engine:latest
```

```bash
curl http://127.0.0.1:8000/health
```

Notebook smoke validation in Docker:

```bash
docker run --rm \
  -v "$(pwd)/artifacts:/app/artifacts" \
  ft-config-engine:latest \
  python /app/scripts/validate_notebooks.py --dataset /app/finetuning_configs_final.jsonl
```

With API key enabled:

```bash
curl -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -X POST http://127.0.0.1:8000/recommend \
  -d '{"platform":"colab","plan":"free","task_type":"instruction_following","adapter_type":"qlora","model_size_bucket":"medium","strategy":"auto","include_notebook":false}'
```

## Full Stack (Backend + Postgres + Frontend)

```bash
docker compose up --build
```

Services:

- Backend API: `http://localhost:8000`
- Frontend UI: `http://localhost:3000`
- Postgres: `localhost:5432`

Optional secure local run:

```bash
API_KEY=dev-local-key REQUIRE_API_KEY=true docker compose up --build
```

Notebook smoke validation via compose:

```bash
docker compose run --rm backend python /app/scripts/validate_notebooks.py
```

## Frontend (Next.js + Tailwind)

Frontend source is in `frontend/`.

```bash
cd frontend
cp .env.example .env.local
# set API_BASE_URL_SERVER and NEXT_PUBLIC_API_BASE_URL if needed
# optionally set API_KEY_SERVER / NEXT_PUBLIC_API_KEY when backend auth is enabled
npm install
npm run dev
```

## Deployment Manifests

- Fly.io: `fly.toml`
- Render: `render.yaml`
- Railway: `railway.json`
- Procfile: `Procfile`
