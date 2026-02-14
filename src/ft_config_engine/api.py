import os
import random
import secrets
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from .db import EngineStore, resolve_database_url
from .governance import DEFAULT_MODEL_NAME, GovernanceError, ModelGovernanceService
from .metrics import CONTENT_TYPE_LATEST, create_metrics_registry
from .models import RecommendationRequest
from .normalization import canonicalize_task_type
from .recommender import build_engine_from_dataset
from .retraining import RetrainingScheduler

try:
    from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
    from pydantic import BaseModel, ConfigDict, Field
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "FastAPI, Pydantic, and SlowAPI are required for API mode. Install requirements.txt first."
    ) from exc


class RecommendRequestModel(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
        extra="forbid",
        str_strip_whitespace=True,
    )

    platform: Literal["colab", "kaggle", "lightning"]
    plan: Literal["free", "pro"]
    task_type: str = Field(..., min_length=1, max_length=128)
    adapter_type: Literal["none", "lora", "qlora"]
    model_size_bucket: Literal["small", "medium", "large"] | None = None
    model_name: str | None = Field(default=None, max_length=256)
    model_parameter_count: str | None = Field(default=None, max_length=32)
    dataset_name: str | None = Field(default=None, max_length=256)
    dataset_size: int | None = Field(default=None, ge=1, le=1_000_000_000)
    sequence_length: int | None = Field(default=None, ge=128, le=8192)
    num_gpus: int | None = Field(default=None, ge=1, le=32)
    gpu_override: str | None = Field(default=None, max_length=64)
    epochs: float | None = Field(default=None, gt=0.0, le=100.0)
    push_to_hub: bool = False
    huggingface_repo_id: str | None = Field(default=None, max_length=256)
    strategy: Literal["auto", "hybrid", "deterministic", "hybrid_ml"] = "auto"
    rerank_top_k: int = Field(default=5, ge=1, le=20)
    include_notebook: bool = True


class BootstrapResponseModel(BaseModel):
    dataset_path: str
    database_url: str
    sync_stats: dict[str, int]
    normalized_configs: int
    profiles: int


class FeedbackRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    recommendation_event_id: int = Field(..., ge=1)
    rating: int | None = Field(default=None, ge=1, le=5)
    success: bool | None = None
    notes: str | None = Field(default=None, max_length=1000)


class TelemetryRunStartRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, protected_namespaces=())

    recommendation_event_id: int | None = Field(default=None, ge=1)
    schema_version: str = Field(default="v1", max_length=8)
    model_id: str = Field(..., min_length=1, max_length=256)
    task_type: str = Field(..., min_length=1, max_length=64)
    adapter_type: Literal["none", "lora", "qlora"]
    dataset_name: str = Field(..., min_length=1, max_length=512)
    dataset_size: int | None = Field(default=None, ge=0, le=10_000_000_000)
    gpu_type: str | None = Field(default=None, max_length=64)
    actual_lr: float | None = Field(default=None, ge=1e-9, le=1.0)
    actual_batch_size: int | None = Field(default=None, ge=1, le=4096)
    actual_gradient_accum: int | None = Field(default=None, ge=1, le=512)
    actual_lora_r: int | None = Field(default=None, ge=1, le=512)
    actual_epochs: float | None = Field(default=None, ge=0.0, le=10_000.0)
    actual_max_steps: int | None = Field(default=None, ge=0)
    was_config_modified: bool = False
    recommendation_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    fallback_level: str | None = Field(default=None, max_length=32)


class TelemetryRunCompleteRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, protected_namespaces=())

    recommendation_event_id: int | None = Field(default=None, ge=1)
    run_start_id: str | None = Field(default=None, max_length=36)
    outcome: Literal["success", "oom", "diverged", "stagnated", "timeout", "user_stopped"]
    final_train_loss: float | None = None
    final_eval_loss: float | None = None
    primary_metric_name: str | None = Field(default=None, max_length=64)
    primary_metric_value: float | None = None
    failure_mode: str | None = Field(default=None, max_length=64)
    failure_step: int | None = Field(default=None, ge=0)
    failure_message: str | None = Field(default=None, max_length=5000)
    wall_clock_minutes: float | None = Field(default=None, ge=0.0, le=100_000.0)
    peak_vram_gb: float | None = Field(default=None, ge=0.0, le=500.0)
    tokens_per_second: float | None = Field(default=None, ge=0.0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0, le=100_000.0)
    user_rating: int | None = Field(default=None, ge=1, le=5)
    user_note: str | None = Field(default=None, max_length=5000)


class PromoteModelRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, protected_namespaces=())

    model_name: str = Field(default=DEFAULT_MODEL_NAME, max_length=128)
    version: str = Field(..., min_length=1, max_length=64)
    to_state: Literal["CANDIDATE", "STAGING", "CANARY", "PRODUCTION", "DEPRECATED", "REJECTED"]
    actor: str = Field(default="manual_operator", max_length=128)
    reason: str | None = Field(default=None, max_length=2000)
    automated: bool = False


class RollbackRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, protected_namespaces=())

    model_name: str = Field(default=DEFAULT_MODEL_NAME, max_length=128)
    to_version: str | None = Field(default=None, max_length=64)
    actor: str = Field(default="on_call_engineer", max_length=128)
    reason: str = Field(default="emergency_rollback", max_length=2000)


def _dataset_path() -> Path:
    return Path(
        os.environ.get("FT_CONFIG_DATASET_PATH", "finetuning_configs_final.jsonl"),
    ).resolve()


def _ml_reranker_path() -> Path | None:
    raw = os.environ.get("FT_CONFIG_ML_RERANKER_PATH", "artifacts/ml_reranker.joblib").strip()
    if not raw:
        return None
    return Path(raw).resolve()


def _hp_predictor_path() -> Path | None:
    raw = os.environ.get("FT_CONFIG_HP_PREDICTOR_PATH", "artifacts/hp_predictor.joblib").strip()
    if not raw:
        return None
    return Path(raw).resolve()


def _canary_ml_reranker_path() -> Path | None:
    raw = os.environ.get("FT_CONFIG_ML_RERANKER_CANARY_PATH", "").strip()
    if not raw:
        return None
    return Path(raw).resolve()


def _canary_hp_predictor_path() -> Path | None:
    raw = os.environ.get("FT_CONFIG_HP_PREDICTOR_CANARY_PATH", "").strip()
    if not raw:
        return None
    return Path(raw).resolve()


def _model_registry_path() -> Path:
    raw = os.environ.get("MODEL_REGISTRY_PATH", "artifacts/model_registry.json").strip()
    return Path(raw).resolve()


def _retraining_state_path() -> Path:
    raw = os.environ.get("RETRAINING_STATE_PATH", "/tmp/ft_config_retraining_state.json").strip()
    return Path(raw).resolve()


def _safe_database_url(url: str) -> str:
    if "@" not in url:
        return url
    scheme_sep = "://"
    if scheme_sep not in url:
        return url
    scheme, rest = url.split(scheme_sep, 1)
    if "@" not in rest:
        return url
    return f"{scheme}{scheme_sep}***:***@{rest.split('@', 1)[1]}"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    fallback = [item.strip() for item in default.split(",") if item.strip()]
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or fallback


API_KEY = os.environ.get("API_KEY", "").strip()
REQUIRE_API_KEY = _env_bool("REQUIRE_API_KEY", default=bool(API_KEY))
EXPOSE_INTERNALS = _env_bool("EXPOSE_INTERNALS", default=False)
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))
HEALTH_RATE_LIMIT = os.environ.get("HEALTH_RATE_LIMIT", "120/minute")
BOOTSTRAP_RATE_LIMIT = os.environ.get("BOOTSTRAP_RATE_LIMIT", "10/minute")
READ_RATE_LIMIT = os.environ.get("READ_RATE_LIMIT", "60/minute")
RECENT_RATE_LIMIT = os.environ.get("RECENT_RATE_LIMIT", "30/minute")
RECOMMEND_RATE_LIMIT = os.environ.get("RECOMMEND_RATE_LIMIT", "40/minute")
FEEDBACK_RATE_LIMIT = os.environ.get("FEEDBACK_RATE_LIMIT", "40/minute")
TELEMETRY_RATE_LIMIT = os.environ.get("TELEMETRY_RATE_LIMIT", "60/minute")
GOVERNANCE_RATE_LIMIT = os.environ.get("GOVERNANCE_RATE_LIMIT", "30/minute")
RETRAIN_RATE_LIMIT = os.environ.get("RETRAIN_RATE_LIMIT", "20/minute")
CANARY_FRACTION = max(0.0, min(1.0, float(os.environ.get("CANARY_FRACTION", "0.0"))))
AUTO_RETRAIN_ON_TELEMETRY = _env_bool("AUTO_RETRAIN_ON_TELEMETRY", default=False)
RETRAIN_CHECK_ON_TELEMETRY = _env_bool("RETRAIN_CHECK_ON_TELEMETRY", default=True)


def _token_from_authorization_header(header_value: str | None) -> str | None:
    if not header_value:
        return None
    prefix = "bearer "
    value = header_value.strip()
    if value.lower().startswith(prefix):
        return value[len(prefix) :].strip()
    return None


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> None:
    if not REQUIRE_API_KEY:
        return
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key authentication is enabled but API_KEY is not configured",
        )
    provided = x_api_key or _token_from_authorization_header(authorization)
    if not provided or not secrets.compare_digest(provided, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )


engine = build_engine_from_dataset(
    _dataset_path(),
    ml_reranker_path=_ml_reranker_path(),
    hp_predictor_path=_hp_predictor_path(),
)
canary_engine = None
if _canary_ml_reranker_path() is not None or _canary_hp_predictor_path() is not None:
    canary_engine = build_engine_from_dataset(
        _dataset_path(),
        ml_reranker_path=_canary_ml_reranker_path(),
        hp_predictor_path=_canary_hp_predictor_path(),
    )
store = EngineStore(resolve_database_url())
store.create_tables()
sync_stats = store.sync_reference_data(engine.configs, engine.profiles)
governance = ModelGovernanceService(store, registry_path=_model_registry_path())
retraining_scheduler = RetrainingScheduler(
    store=store,
    governance=governance,
    state_path=_retraining_state_path(),
)
metrics_registry = create_metrics_registry()

app = FastAPI(title="Fine-Tune Configuration Engine", version="0.1.0")
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{RATE_LIMIT_PER_MINUTE}/minute"],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

if _env_bool("FORCE_HTTPS", default=False):
    app.add_middleware(HTTPSRedirectMiddleware)

trusted_hosts = _env_csv("TRUSTED_HOSTS", "localhost,127.0.0.1")
if trusted_hosts != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_env_csv("CORS_ORIGINS", "http://localhost:3000"),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def canary_slot_middleware(request: Request, call_next):
    request.state.model_slot = "production"
    request.state.model_version = (
        engine.ml_reranker.model_version if engine.ml_reranker is not None else "deterministic"
    )
    if request.url.path == "/recommend" and canary_engine is not None and CANARY_FRACTION > 0.0:
        if random.random() < CANARY_FRACTION:
            request.state.model_slot = "canary"
            request.state.model_version = (
                canary_engine.ml_reranker.model_version if canary_engine.ml_reranker is not None else "deterministic"
            )
    response = await call_next(request)
    response.headers.setdefault("X-Model-Slot", str(getattr(request.state, "model_slot", "production")))
    response.headers.setdefault("X-Model-Version", str(getattr(request.state, "model_version", "unknown")))
    return response


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'; base-uri 'none'")
    if request.url.scheme == "https":
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


@app.get("/health")
@limiter.limit(HEALTH_RATE_LIMIT)
def health(request: Request) -> dict:
    ml_metadata = engine.ml_reranker.metadata.to_dict() if engine.ml_reranker else None
    hp_metadata = engine.hp_predictor.metadata.to_dict() if engine.hp_predictor else None
    return {
        "status": "ok",
        "dataset_path": str(_dataset_path()) if EXPOSE_INTERNALS else "hidden",
        "database_url": _safe_database_url(store.database_url) if EXPOSE_INTERNALS else "hidden",
        "normalized_configs": len(engine.configs),
        "profiles": len(engine.profiles),
        "ml_reranker_loaded": engine.ml_reranker is not None,
        "ml_reranker_status": engine.ml_reranker_status,
        "ml_reranker_model_version": engine.ml_reranker.model_version if engine.ml_reranker else None,
        "ml_reranker_metrics": (ml_metadata.get("metrics", {}) if (EXPOSE_INTERNALS and ml_metadata) else {}),
        "hp_predictor_loaded": engine.hp_predictor is not None,
        "hp_predictor_status": engine.hp_predictor_status,
        "hp_predictor_model_version": engine.hp_predictor.model_version if engine.hp_predictor else None,
        "hp_predictor_metrics": (
            hp_metadata.get("target_metrics", {}) if (EXPOSE_INTERNALS and hp_metadata) else {}
        ),
        "sync_stats": sync_stats if EXPOSE_INTERNALS else {},
        "auth_required": REQUIRE_API_KEY,
        "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
        "canary_enabled": canary_engine is not None and CANARY_FRACTION > 0.0,
        "canary_fraction": CANARY_FRACTION,
        "prometheus_metrics_enabled": metrics_registry.enabled,
    }


@app.get("/ml-reranker")
@limiter.limit(READ_RATE_LIMIT)
def ml_reranker_info(
    request: Request,
    _: None = Depends(require_api_key),
) -> dict:
    if engine.ml_reranker is None:
        return {
            "loaded": False,
            "status": engine.ml_reranker_status,
            "model_version": None,
            "metadata": None,
        }

    metadata = engine.ml_reranker.metadata.to_dict()
    if not EXPOSE_INTERNALS:
        metadata = {
            "model_version": metadata.get("model_version"),
            "trained_at_utc": metadata.get("trained_at_utc"),
            "metrics": metadata.get("metrics", {}),
        }

    return {
        "loaded": True,
        "status": engine.ml_reranker_status,
        "model_version": engine.ml_reranker.model_version,
        "metadata": metadata,
    }


@app.get("/hp-predictor")
@limiter.limit(READ_RATE_LIMIT)
def hp_predictor_info(
    request: Request,
    _: None = Depends(require_api_key),
) -> dict:
    if engine.hp_predictor is None:
        return {
            "loaded": False,
            "status": engine.hp_predictor_status,
            "model_version": None,
            "metadata": None,
        }

    metadata = engine.hp_predictor.metadata.to_dict()
    if not EXPOSE_INTERNALS:
        metadata = {
            "model_version": metadata.get("model_version"),
            "trained_at_utc": metadata.get("trained_at_utc"),
            "trained_targets": metadata.get("trained_targets", []),
            "target_metrics": metadata.get("target_metrics", {}),
        }
    return {
        "loaded": True,
        "status": engine.hp_predictor_status,
        "model_version": engine.hp_predictor.model_version,
        "metadata": metadata,
    }


@app.post("/bootstrap", response_model=BootstrapResponseModel)
@limiter.limit(BOOTSTRAP_RATE_LIMIT)
def bootstrap(
    request: Request,
    _: None = Depends(require_api_key),
) -> BootstrapResponseModel:
    global engine, canary_engine, sync_stats
    refreshed_engine = build_engine_from_dataset(
        _dataset_path(),
        ml_reranker_path=_ml_reranker_path(),
        hp_predictor_path=_hp_predictor_path(),
    )
    refreshed_canary_engine = None
    if _canary_ml_reranker_path() is not None or _canary_hp_predictor_path() is not None:
        refreshed_canary_engine = build_engine_from_dataset(
            _dataset_path(),
            ml_reranker_path=_canary_ml_reranker_path(),
            hp_predictor_path=_canary_hp_predictor_path(),
        )
    refreshed_stats = store.sync_reference_data(refreshed_engine.configs, refreshed_engine.profiles)
    engine = refreshed_engine
    canary_engine = refreshed_canary_engine
    sync_stats = refreshed_stats
    return BootstrapResponseModel(
        dataset_path=str(_dataset_path()),
        database_url=_safe_database_url(store.database_url),
        sync_stats=refreshed_stats,
        normalized_configs=len(refreshed_engine.configs),
        profiles=len(refreshed_engine.profiles),
    )


@app.get("/profiles")
@limiter.limit(READ_RATE_LIMIT)
def profiles(
    request: Request,
    _: None = Depends(require_api_key),
    task_type: str | None = None,
    model_size_bucket: str | None = None,
    adapter_type: str | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict:
    rows = store.list_profiles(
        task_type=task_type,
        model_size_bucket=model_size_bucket,
        adapter_type=adapter_type,
        limit=limit,
    )
    return {"count": len(rows), "items": rows}


@app.get("/configs")
@limiter.limit(READ_RATE_LIMIT)
def configs(
    request: Request,
    _: None = Depends(require_api_key),
    task_type: str | None = None,
    model_size_bucket: str | None = None,
    adapter_type: str | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict:
    rows = store.list_configs(
        task_type=task_type,
        model_size_bucket=model_size_bucket,
        adapter_type=adapter_type,
        limit=limit,
    )
    return {"count": len(rows), "items": rows}


@app.get("/recommendations/recent")
@limiter.limit(RECENT_RATE_LIMIT)
def recommendations_recent(
    request: Request,
    _: None = Depends(require_api_key),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict:
    rows = store.list_recommendations(limit=limit)
    return {"count": len(rows), "items": rows}


@app.get("/brain/strategy")
@limiter.limit(READ_RATE_LIMIT)
def brain_strategy(
    request: Request,
    _: None = Depends(require_api_key),
    task_type: str = Query(..., min_length=1),
    adapter_type: Literal["none", "lora", "qlora"] = Query(...),
    model_size_bucket: Literal["small", "medium", "large"] | None = Query(default=None),
) -> dict:
    resolved_task = canonicalize_task_type(task_type)
    decision = store.learned_strategy_for_context(
        task_type=resolved_task,
        adapter_type=adapter_type,
        model_size_bucket=model_size_bucket,
    )
    return {
        "requested_task_type": task_type,
        "resolved_task_type": resolved_task,
        "decision": decision,
    }


@app.post("/feedback")
@limiter.limit(FEEDBACK_RATE_LIMIT)
def recommendation_feedback(
    request: Request,
    payload: FeedbackRequestModel,
    _: None = Depends(require_api_key),
) -> dict:
    if payload.rating is None and payload.success is None and not (payload.notes or "").strip():
        raise HTTPException(status_code=422, detail="at least one of rating/success/notes is required")
    try:
        saved = store.submit_feedback(
            recommendation_event_id=payload.recommendation_event_id,
            rating=payload.rating,
            success=payload.success,
            notes=payload.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "ok", "feedback": saved}


@app.post("/telemetry/run-start")
@limiter.limit(TELEMETRY_RATE_LIMIT)
def telemetry_run_start(
    request: Request,
    payload: TelemetryRunStartRequestModel,
    _: None = Depends(require_api_key),
) -> dict:
    try:
        saved = store.record_telemetry_run_start(payload.model_dump())
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if detail.endswith("_not_found") else 422
        raise HTTPException(status_code=status_code, detail=detail) from exc
    metrics_registry.inc_telemetry(event_type="run_start", outcome="n/a")
    return {"status": "ok", "run_start": saved}


@app.post("/telemetry/run-complete")
@limiter.limit(TELEMETRY_RATE_LIMIT)
def telemetry_run_complete(
    request: Request,
    payload: TelemetryRunCompleteRequestModel,
    _: None = Depends(require_api_key),
) -> dict:
    try:
        saved = store.record_telemetry_run_complete(payload.model_dump())
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if detail.endswith("_not_found") else 422
        raise HTTPException(status_code=status_code, detail=detail) from exc
    outcome = str(payload.outcome)
    metrics_registry.inc_telemetry(event_type="run_complete", outcome=outcome)
    if outcome == "oom":
        metrics_registry.inc_oom_violation(
            gpu_type="unknown",
            adapter_type="unknown",
            model_size_bucket="unknown",
        )
    snapshot = store.telemetry_trigger_snapshot(
        window_start=datetime.now(timezone.utc) - timedelta(days=7)
    )
    metrics_registry.set_telemetry_backlog(int(snapshot["unprocessed_events"]))

    retrain_decision: dict | None = None
    retrain_result: dict | None = None
    if RETRAIN_CHECK_ON_TELEMETRY:
        retrain_decision = retraining_scheduler.evaluate().to_dict()
        if AUTO_RETRAIN_ON_TELEMETRY and retrain_decision.get("should_retrain"):
            retrain_result = retraining_scheduler.trigger_if_needed(
                dataset=str(_dataset_path()),
                run_training=True,
            )
    return {
        "status": "ok",
        "run_complete": saved,
        "retraining_decision": retrain_decision,
        "retraining_result": retrain_result,
    }


@app.get("/telemetry/recent")
@limiter.limit(READ_RATE_LIMIT)
def telemetry_recent(
    request: Request,
    _: None = Depends(require_api_key),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict:
    items = store.list_telemetry_runs(limit=limit)
    return {
        "run_start_count": len(items["run_start"]),
        "run_complete_count": len(items["run_complete"]),
        "items": items,
    }


@app.get("/metrics")
@limiter.limit(READ_RATE_LIMIT)
def metrics(
    request: Request,
    _: None = Depends(require_api_key),
) -> Response:
    if not metrics_registry.enabled:
        raise HTTPException(status_code=501, detail="prometheus_client_not_installed")
    snapshot = store.telemetry_trigger_snapshot(
        window_start=datetime.now(timezone.utc) - timedelta(days=7)
    )
    metrics_registry.set_telemetry_backlog(int(snapshot["unprocessed_events"]))
    decision = retraining_scheduler.evaluate().to_dict()
    last_train_raw = decision.get("last_trained_at")
    if isinstance(last_train_raw, str):
        try:
            last_train = datetime.fromisoformat(last_train_raw)
            lag_hours = (datetime.now(timezone.utc) - last_train).total_seconds() / 3600.0
            metrics_registry.set_retrain_lag_hours(lag_hours)
        except Exception:
            pass
    payload = metrics_registry.export_payload()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


@app.get("/governance/events")
@limiter.limit(GOVERNANCE_RATE_LIMIT)
def governance_events(
    request: Request,
    _: None = Depends(require_api_key),
    model_name: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict:
    items = store.list_governance_events(model_name=model_name, limit=limit)
    return {"count": len(items), "items": items}


@app.get("/governance/model")
@limiter.limit(GOVERNANCE_RATE_LIMIT)
def governance_model_state(
    request: Request,
    _: None = Depends(require_api_key),
    model_name: str = Query(default=DEFAULT_MODEL_NAME),
) -> dict:
    return governance.list_versions(model_name)


@app.get("/governance/canary/summary")
@limiter.limit(GOVERNANCE_RATE_LIMIT)
def governance_canary_summary(
    request: Request,
    _: None = Depends(require_api_key),
    lookback: int = Query(default=500, ge=1, le=10000),
) -> dict:
    summary = store.canary_rollout_summary(lookback=lookback)
    return {"status": "ok", "summary": summary}


@app.post("/governance/promote")
@limiter.limit(GOVERNANCE_RATE_LIMIT)
def governance_promote(
    request: Request,
    payload: PromoteModelRequestModel,
    _: None = Depends(require_api_key),
) -> dict:
    try:
        result = governance.promote(
            model_name=payload.model_name,
            version=payload.version,
            to_state=payload.to_state,
            actor=payload.actor,
            reason=payload.reason or "",
            automated=payload.automated,
        )
    except GovernanceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    metrics_registry.inc_governance_transition(
        model_name=result.model_name,
        from_state=result.from_state,
        to_state=result.to_state,
        automated=result.automated,
    )
    return {"status": "ok", "promotion": result.to_dict()}


@app.post("/governance/rollback")
@limiter.limit(GOVERNANCE_RATE_LIMIT)
def governance_rollback(
    request: Request,
    payload: RollbackRequestModel,
    _: None = Depends(require_api_key),
) -> dict:
    try:
        result = governance.rollback(
            model_name=payload.model_name,
            to_version=payload.to_version,
            actor=payload.actor,
            reason=payload.reason,
        )
    except GovernanceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"status": "ok", "rollback": result}


@app.get("/retraining/status")
@limiter.limit(RETRAIN_RATE_LIMIT)
def retraining_status(
    request: Request,
    _: None = Depends(require_api_key),
) -> dict:
    decision = retraining_scheduler.evaluate()
    return {"status": "ok", "decision": decision.to_dict()}


@app.post("/retraining/check")
@limiter.limit(RETRAIN_RATE_LIMIT)
def retraining_check(
    request: Request,
    _: None = Depends(require_api_key),
    run: bool = Query(default=False),
) -> dict:
    outcome = retraining_scheduler.trigger_if_needed(dataset=str(_dataset_path()), run_training=run)
    return {"status": "ok", "result": outcome}


@app.post("/recommend")
@limiter.limit(RECOMMEND_RATE_LIMIT)
def recommend(
    request: Request,
    response: Response,
    payload: RecommendRequestModel,
    _: None = Depends(require_api_key),
) -> dict:
    req = RecommendationRequest(
        platform=payload.platform,
        plan=payload.plan,
        task_type=payload.task_type,
        adapter_type=payload.adapter_type,
        model_size_bucket=payload.model_size_bucket,
        model_name=payload.model_name,
        model_parameter_count=payload.model_parameter_count,
        dataset_name=payload.dataset_name,
        dataset_size=payload.dataset_size,
        sequence_length=payload.sequence_length,
        num_gpus=payload.num_gpus,
        gpu_override=payload.gpu_override,
        epochs=payload.epochs,
        push_to_hub=payload.push_to_hub,
        huggingface_repo_id=payload.huggingface_repo_id,
        strategy=payload.strategy,
        rerank_top_k=payload.rerank_top_k,
    )
    active_engine = canary_engine if getattr(request.state, "model_slot", "production") == "canary" and canary_engine else engine

    brain_decision: dict = {}
    brain_strategy_hint: str | None = None
    if payload.strategy == "auto":
        resolved_task = canonicalize_task_type(payload.task_type)
        brain_decision = store.learned_strategy_for_context(
            task_type=resolved_task,
            adapter_type=payload.adapter_type,
            model_size_bucket=payload.model_size_bucket,
        )
        candidate_hint = brain_decision.get("strategy")
        if candidate_hint in {"deterministic", "hybrid", "hybrid_ml"}:
            brain_strategy_hint = str(candidate_hint)
    try:
        with metrics_registry.latency_timer(payload.strategy):
            result = active_engine.recommend(
                req,
                render_notebook=payload.include_notebook,
                brain_strategy_hint=brain_strategy_hint,
                brain_decision=brain_decision,
            )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="invalid recommendation request") from exc
    payload_dict = result.to_dict()
    basis = payload_dict.get("recommendation_basis", {})
    basis["model_slot"] = str(getattr(request.state, "model_slot", "production"))
    basis["model_version"] = str(getattr(request.state, "model_version", "unknown"))
    request_payload = payload.model_dump()
    request_payload["_model_slot"] = basis["model_slot"]
    request_payload["_model_version"] = basis["model_version"]

    event_id = store.log_recommendation(request_payload=request_payload, result_payload=payload_dict)
    payload_dict["recommendation_event_id"] = event_id
    metrics_registry.inc_recommendation(
        task_type=basis.get("resolved_task_type") or payload.task_type,
        adapter_type=basis.get("resolved_adapter_type") or payload.adapter_type,
        confidence_level=basis.get("confidence_level") or "unknown",
        strategy=basis.get("strategy") or payload.strategy,
        model_slot=basis.get("model_slot") or "production",
    )
    metrics_registry.observe_confidence(
        task_type=basis.get("resolved_task_type") or payload.task_type,
        fallback_level=basis.get("profile_match_type") or "unknown",
        confidence_score=float(basis.get("confidence_score") or 0.0),
    )
    response.headers.setdefault("X-Model-Slot", str(basis["model_slot"]))
    response.headers.setdefault("X-Model-Version", str(basis["model_version"]))
    return payload_dict
