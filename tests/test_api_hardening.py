from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "finetuning_configs_final.jsonl"


def _payload() -> dict:
    return {
        "platform": "colab",
        "plan": "free",
        "task_type": "instruction_following",
        "adapter_type": "qlora",
        "model_size_bucket": "medium",
        "sequence_length": 1024,
        "num_gpus": 1,
        "epochs": 1.0,
        "strategy": "deterministic",
        "include_notebook": False,
    }


def _load_api_module(monkeypatch, tmp_path: Path, **overrides):
    defaults: dict[str, str] = {
        "FT_CONFIG_DATASET_PATH": str(DATASET_PATH),
        "DATABASE_URL": f"sqlite:///{(tmp_path / 'ft_config_engine_test.db').resolve()}",
        "MODEL_REGISTRY_PATH": str((tmp_path / "model_registry.json").resolve()),
        "TRUSTED_HOSTS": "*",
        "CORS_ORIGINS": "http://localhost:3000",
        "EXPOSE_INTERNALS": "false",
        "REQUIRE_API_KEY": "false",
        "RECOMMEND_RATE_LIMIT": "40/minute",
    }
    defaults.update(overrides)

    for key, value in defaults.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
            continue
        monkeypatch.setenv(key, value)

    module_name = "ft_config_engine.api"
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def test_health_includes_security_headers(monkeypatch, tmp_path):
    api_module = _load_api_module(monkeypatch, tmp_path)
    client = TestClient(api_module.app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["dataset_path"] == "hidden"
    assert response.json()["ml_reranker_metrics"] == {}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'none'" in response.headers["content-security-policy"]


def test_recommend_requires_api_key_when_enabled(monkeypatch, tmp_path):
    api_module = _load_api_module(
        monkeypatch,
        tmp_path,
        REQUIRE_API_KEY="true",
        API_KEY="super-secret-key",
    )
    client = TestClient(api_module.app)

    unauthorized = client.post("/recommend", json=_payload())
    assert unauthorized.status_code == 401

    authorized = client.post(
        "/recommend",
        json=_payload(),
        headers={"X-API-Key": "super-secret-key"},
    )
    assert authorized.status_code == 200
    assert authorized.json()["recommendation_basis"]["strategy"] == "deterministic"


def test_recommend_validates_sequence_length(monkeypatch, tmp_path):
    api_module = _load_api_module(monkeypatch, tmp_path)
    client = TestClient(api_module.app)

    bad_payload = _payload()
    bad_payload["sequence_length"] = 32
    response = client.post("/recommend", json=bad_payload)

    assert response.status_code == 422


def test_recommend_rate_limit(monkeypatch, tmp_path):
    api_module = _load_api_module(
        monkeypatch,
        tmp_path,
        RECOMMEND_RATE_LIMIT="1/minute",
    )
    client = TestClient(api_module.app)

    first = client.post("/recommend", json=_payload())
    assert first.status_code == 200

    second = client.post("/recommend", json=_payload())
    assert second.status_code == 429


def test_task_alias_is_normalized(monkeypatch, tmp_path):
    api_module = _load_api_module(monkeypatch, tmp_path)
    client = TestClient(api_module.app)

    payload = _payload()
    payload["task_type"] = "language finetuning"
    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["recommendation_basis"]["resolved_task_type"] == "instruction_following"
    assert any("task normalized" in note for note in body["notes"])


def test_ocr_alias_uses_task_pool_and_adjusts_adapter(monkeypatch, tmp_path):
    api_module = _load_api_module(monkeypatch, tmp_path)
    client = TestClient(api_module.app)

    payload = _payload()
    payload["task_type"] = "ocr_finetuning"
    payload["adapter_type"] = "qlora"
    payload.pop("sequence_length", None)
    payload.pop("num_gpus", None)
    payload.pop("epochs", None)
    payload["strategy"] = "auto"
    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["recommendation_basis"]["resolved_task_type"] == "optical_character_recognition"
    assert body["recommendation_basis"]["candidate_pool_size"] > 0
    assert body["recommendation_basis"]["resolved_adapter_type"] == "none"
    assert body["safe_hyperparameters"]["adapter_type"] == "none"
    assert body["recommendation_basis"]["model_size_bucket"] == "small"
    assert any("adapter adjusted" in note for note in body["notes"])


def test_unknown_task_uses_low_confidence_guard(monkeypatch, tmp_path):
    api_module = _load_api_module(monkeypatch, tmp_path)
    client = TestClient(api_module.app)

    payload = _payload()
    payload["task_type"] = "my_custom_task"
    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["recommendation_basis"]["confidence_level"] == "low"
    assert body["recommendation_basis"]["candidate_pool_size"] == 0
    assert body["safe_hyperparameters"]["batch_size_per_device"] <= 8
    assert body["safe_hyperparameters"]["max_seq_length"] <= 768


def test_corrupted_additional_dataset_file_is_ignored(monkeypatch, tmp_path):
    corrupted_dataset = tmp_path / "real_world_corrupted.jsonl"
    corrupted_dataset.write_bytes(b"\x00" * 1024)
    dataset_paths = f"{DATASET_PATH},{corrupted_dataset}"

    api_module = _load_api_module(monkeypatch, tmp_path, FT_CONFIG_DATASET_PATH=dataset_paths)
    client = TestClient(api_module.app)

    response = client.post("/recommend", json=_payload())
    assert response.status_code == 200
    body = response.json()

    reasons = body["recommendation_basis"]["normalization_report"]["rejection_reasons"]
    assert any(key.startswith("dataset_file_null_bytes:") for key in reasons)


def test_real_world_style_schema_is_normalized(monkeypatch, tmp_path):
    real_world_dataset = tmp_path / "real_world_style.jsonl"
    row = {
        "id": "unit_real_world_001",
        "task_type": "classification",
        "dataset": {"name": "unit_dataset", "size": 12345, "format": "hf_dataset"},
        "model": {
            "name": "bert-base-uncased",
            "size_params": "110M",
            "architecture": "BertForSequenceClassification",
        },
        "training_config": {
            "learning_rate": 2e-5,
            "num_epochs": 3,
            "batch_size": 16,
        },
        "hardware_config": {"target_gpu": "T4", "gpu_memory_gb": 16, "quantization": "none"},
        "source": "unit_test",
    }
    real_world_dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")

    dataset_paths = f"{DATASET_PATH},{real_world_dataset}"
    api_module = _load_api_module(monkeypatch, tmp_path, FT_CONFIG_DATASET_PATH=dataset_paths)

    ingested = next(cfg for cfg in api_module.engine.configs if cfg.record_id == "unit_real_world_001")
    assert ingested.task_type == "classification"
    assert ingested.model_parameter_count_num == 110_000_000
    assert ingested.gradient_accumulation_steps == 1
    assert ingested.max_seq_length == 512
    assert ingested.gpu_type == "T4"

    reasons = api_module.engine.normalization_report.rejection_reasons
    assert reasons.get("missing_task_type", 0) == 0


def test_hybrid_ml_falls_back_when_model_artifact_missing(monkeypatch, tmp_path):
    missing_model = tmp_path / "missing_ml_reranker.joblib"
    api_module = _load_api_module(
        monkeypatch,
        tmp_path,
        FT_CONFIG_ML_RERANKER_PATH=str(missing_model),
    )
    client = TestClient(api_module.app)

    payload = _payload()
    payload["strategy"] = "hybrid_ml"
    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["recommendation_basis"]["requested_strategy"] == "hybrid_ml"
    assert body["recommendation_basis"]["strategy"] == "hybrid"
    assert body["recommendation_basis"]["ml_reranker"]["loaded"] is False
    assert any("fallback to hybrid" in note for note in body["notes"])


def test_auto_strategy_falls_back_to_hybrid_when_ml_missing(monkeypatch, tmp_path):
    missing_model = tmp_path / "missing_ml_reranker.joblib"
    api_module = _load_api_module(
        monkeypatch,
        tmp_path,
        FT_CONFIG_ML_RERANKER_PATH=str(missing_model),
    )
    client = TestClient(api_module.app)

    payload = _payload()
    payload.pop("strategy", None)
    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["recommendation_basis"]["requested_strategy"] == "auto"
    assert body["recommendation_basis"]["strategy"] == "hybrid"
    assert any("strategy auto-selected" in note for note in body["notes"])


def test_auto_controls_are_resolved_by_engine(monkeypatch, tmp_path):
    missing_model = tmp_path / "missing_ml_reranker.joblib"
    api_module = _load_api_module(
        monkeypatch,
        tmp_path,
        FT_CONFIG_ML_RERANKER_PATH=str(missing_model),
    )
    client = TestClient(api_module.app)

    payload = _payload()
    payload.pop("sequence_length", None)
    payload.pop("num_gpus", None)
    payload.pop("epochs", None)
    payload.pop("strategy", None)
    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    body = response.json()
    basis = body["recommendation_basis"]
    assert basis["requested_sequence_length"] is None
    assert basis["requested_num_gpus"] is None
    assert basis["requested_epochs"] is None
    assert basis["resolved_sequence_length"] >= 128
    assert basis["resolved_num_gpus"] >= 1
    assert basis["resolved_epochs"] >= 1.0
    assert any("auto-set" in note for note in body["notes"])


def test_ml_reranker_endpoint_reports_missing_model(monkeypatch, tmp_path):
    missing_model = tmp_path / "missing_ml_reranker.joblib"
    api_module = _load_api_module(
        monkeypatch,
        tmp_path,
        FT_CONFIG_ML_RERANKER_PATH=str(missing_model),
    )
    client = TestClient(api_module.app)

    response = client.get("/ml-reranker")
    assert response.status_code == 200
    body = response.json()
    assert body["loaded"] is False
    assert body["model_version"] is None
    assert body["metadata"] is None


def test_hp_predictor_endpoint_reports_missing_model(monkeypatch, tmp_path):
    missing_model = tmp_path / "missing_hp_predictor.joblib"
    api_module = _load_api_module(
        monkeypatch,
        tmp_path,
        FT_CONFIG_HP_PREDICTOR_PATH=str(missing_model),
    )
    client = TestClient(api_module.app)

    response = client.get("/hp-predictor")
    assert response.status_code == 200
    body = response.json()
    assert body["loaded"] is False
    assert body["model_version"] is None
    assert body["metadata"] is None


def test_feedback_guides_auto_strategy(monkeypatch, tmp_path):
    missing_model = tmp_path / "missing_ml_reranker.joblib"
    api_module = _load_api_module(
        monkeypatch,
        tmp_path,
        FT_CONFIG_ML_RERANKER_PATH=str(missing_model),
    )
    client = TestClient(api_module.app)

    deterministic_payload = _payload()
    deterministic_payload["strategy"] = "deterministic"
    deterministic_payload["include_notebook"] = False
    deterministic_response = client.post("/recommend", json=deterministic_payload)
    assert deterministic_response.status_code == 200
    deterministic_event_id = deterministic_response.json()["recommendation_event_id"]

    hybrid_payload = _payload()
    hybrid_payload["strategy"] = "hybrid"
    hybrid_payload["include_notebook"] = False
    hybrid_response = client.post("/recommend", json=hybrid_payload)
    assert hybrid_response.status_code == 200
    hybrid_event_id = hybrid_response.json()["recommendation_event_id"]

    feedback_good = client.post(
        "/feedback",
        json={
            "recommendation_event_id": deterministic_event_id,
            "rating": 5,
            "success": True,
            "notes": "solid run",
        },
    )
    assert feedback_good.status_code == 200

    feedback_bad = client.post(
        "/feedback",
        json={
            "recommendation_event_id": hybrid_event_id,
            "rating": 1,
            "success": False,
            "notes": "not stable",
        },
    )
    assert feedback_bad.status_code == 200

    decision_probe = client.get(
        "/brain/strategy",
        params={
            "task_type": "instruction_following",
            "adapter_type": "qlora",
            "model_size_bucket": "medium",
        },
    )
    assert decision_probe.status_code == 200
    assert decision_probe.json()["decision"]["strategy"] == "deterministic"

    auto_payload = _payload()
    auto_payload.pop("strategy", None)
    auto_response = client.post("/recommend", json=auto_payload)
    assert auto_response.status_code == 200
    auto_body = auto_response.json()
    assert auto_body["recommendation_basis"]["requested_strategy"] == "auto"
    assert auto_body["recommendation_basis"]["strategy"] == "deterministic"
    assert auto_body["recommendation_basis"]["brain_decision"]["strategy"] == "deterministic"
    assert any("learning brain" in note for note in auto_body["notes"])


def test_unknown_task_exposes_ood_guard_signals(monkeypatch, tmp_path):
    api_module = _load_api_module(monkeypatch, tmp_path)
    client = TestClient(api_module.app)

    payload = _payload()
    payload["task_type"] = "totally_unknown_task_family"
    payload["include_notebook"] = False
    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    body = response.json()
    basis = body["recommendation_basis"]
    assert basis["unsupported_task"] is True
    assert basis["ood_guard_active"] is True
    assert basis["template_fallback_warning"] is not None
    assert body["safe_hyperparameters"]["batch_size_per_device"] <= 4
    assert body["safe_hyperparameters"]["max_seq_length"] <= 512


def test_telemetry_endpoints_store_and_return_records(monkeypatch, tmp_path):
    api_module = _load_api_module(monkeypatch, tmp_path)
    client = TestClient(api_module.app)

    rec_response = client.post("/recommend", json=_payload())
    assert rec_response.status_code == 200
    event_id = rec_response.json()["recommendation_event_id"]

    run_start_response = client.post(
        "/telemetry/run-start",
        json={
            "recommendation_event_id": event_id,
            "model_id": "meta-llama/Llama-2-7b-hf",
            "task_type": "instruction_following",
            "adapter_type": "qlora",
            "dataset_name": "alpaca",
            "dataset_size": 52000,
            "gpu_type": "T4",
            "actual_lr": 0.0002,
            "actual_batch_size": 4,
            "actual_gradient_accum": 4,
            "actual_lora_r": 16,
            "recommendation_confidence": 0.82,
        },
    )
    assert run_start_response.status_code == 200
    run_start_id = run_start_response.json()["run_start"]["id"]

    run_complete_response = client.post(
        "/telemetry/run-complete",
        json={
            "recommendation_event_id": event_id,
            "run_start_id": run_start_id,
            "outcome": "success",
            "final_train_loss": 0.42,
            "primary_metric_name": "rougeL",
            "primary_metric_value": 0.31,
            "wall_clock_minutes": 23.4,
            "peak_vram_gb": 13.7,
            "tokens_per_second": 128.2,
            "estimated_cost_usd": 0.47,
            "user_rating": 4,
            "user_note": "works well for my test run",
        },
    )
    assert run_complete_response.status_code == 200

    recent_response = client.get("/telemetry/recent", params={"limit": 10})
    assert recent_response.status_code == 200
    body = recent_response.json()
    assert body["run_start_count"] >= 1
    assert body["run_complete_count"] >= 1
    assert any(item["id"] == run_start_id for item in body["items"]["run_start"])


def test_telemetry_validates_out_of_range_payload(monkeypatch, tmp_path):
    api_module = _load_api_module(monkeypatch, tmp_path)
    client = TestClient(api_module.app)

    response = client.post(
        "/telemetry/run-start",
        json={
            "model_id": "meta-llama/Llama-2-7b-hf",
            "task_type": "instruction_following",
            "adapter_type": "qlora",
            "dataset_name": "alpaca",
            "actual_lr": 1.5,
        },
    )
    assert response.status_code == 422
