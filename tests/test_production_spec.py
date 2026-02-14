from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from ft_config_engine.db import EngineStore
from ft_config_engine.evaluation.harness import run_evaluation
from ft_config_engine.governance import ModelGovernanceService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "finetuning_configs_final.jsonl"


def _write_small_golden(path: Path) -> None:
    rows = [
        {
            "id": "golden_001",
            "category": "classification",
            "difficulty": "easy",
            "request": {
                "task_type": "classification",
                "model_id": "bert-base-uncased",
                "model_params": 110_000_000,
                "adapter_type": "none",
                "dataset_name": "imdb",
                "dataset_size": 25_000,
                "gpu_type": "T4",
                "gpu_vram_gb": 16,
                "platform": "colab",
                "plan": "free",
                "sequence_length": 512,
                "num_gpus": 1,
                "strategy": "deterministic",
            },
            "ground_truth": {
                "learning_rate": 2e-5,
                "per_device_train_batch_size": 16,
                "gradient_accumulation_steps": 1,
                "lora_r": None,
            },
            "acceptable_ranges": {"learning_rate": [1e-5, 1e-4], "batch_size": [8, 32], "lora_r": None},
        },
        {
            "id": "golden_002",
            "category": "causal_lm",
            "difficulty": "medium",
            "request": {
                "task_type": "instruction_following",
                "model_id": "meta-llama/Llama-2-7b-hf",
                "model_params": 7_000_000_000,
                "adapter_type": "qlora",
                "dataset_name": "alpaca",
                "dataset_size": 52_000,
                "gpu_type": "T4",
                "gpu_vram_gb": 16,
                "platform": "colab",
                "plan": "free",
                "sequence_length": 1024,
                "num_gpus": 1,
                "strategy": "deterministic",
            },
            "ground_truth": {
                "learning_rate": 2e-4,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "lora_r": 16,
            },
            "acceptable_ranges": {"learning_rate": [1e-5, 5e-4], "batch_size": [1, 8], "lora_r": [4, 64]},
        },
        {
            "id": "golden_003",
            "category": "ood_synthetic",
            "difficulty": "hard",
            "request": {
                "task_type": "totally_unknown_task_family",
                "model_id": "meta-llama/Llama-2-7b-hf",
                "model_params": 7_000_000_000,
                "adapter_type": "qlora",
                "dataset_name": "unknown_dataset",
                "dataset_size": 10_000,
                "gpu_type": "T4",
                "gpu_vram_gb": 16,
                "platform": "colab",
                "plan": "free",
                "sequence_length": 1024,
                "num_gpus": 1,
                "strategy": "deterministic",
            },
            "ground_truth": {
                "learning_rate": 2e-4,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "lora_r": 16,
            },
            "acceptable_ranges": {"learning_rate": [1e-5, 5e-4], "batch_size": [1, 8], "lora_r": [4, 64]},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _load_api_module(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("FT_CONFIG_DATASET_PATH", str(DATASET_PATH))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'api_prod_spec.db').resolve()}")
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str((tmp_path / "model_registry.json").resolve()))
    monkeypatch.setenv("TRUSTED_HOSTS", "*")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000")
    monkeypatch.setenv("REQUIRE_API_KEY", "false")
    monkeypatch.setenv("CANARY_FRACTION", "0.0")

    module_name = "ft_config_engine.api"
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def test_evaluation_harness_runs_with_small_golden(tmp_path):
    golden_path = tmp_path / "golden_small.jsonl"
    _write_small_golden(golden_path)
    report = run_evaluation(
        candidate_version="unit_test_version",
        dataset_path=DATASET_PATH,
        golden_path=golden_path,
    )
    assert report.sample_count == 3
    assert len(report.gate_results) == 6
    assert isinstance(report.all_gates_passed, bool)
    assert report.per_category_acc["classification"] >= 0.0


def test_governance_local_state_machine_and_rollback(tmp_path):
    db_path = tmp_path / "governance.db"
    store = EngineStore(f"sqlite:///{db_path}")
    store.create_tables()
    governance = ModelGovernanceService(store, registry_path=tmp_path / "registry.json")

    governance.promote(
        model_name="finetune_config_reranker",
        version="100",
        to_state="CANDIDATE",
        actor="tester",
    )
    governance.promote(
        model_name="finetune_config_reranker",
        version="100",
        to_state="STAGING",
        actor="tester",
    )
    governance.promote(
        model_name="finetune_config_reranker",
        version="100",
        to_state="CANARY",
        actor="tester",
    )
    governance.promote(
        model_name="finetune_config_reranker",
        version="100",
        to_state="PRODUCTION",
        actor="tester",
    )
    governance.promote(
        model_name="finetune_config_reranker",
        version="090",
        to_state="CANDIDATE",
        actor="tester",
    )
    governance.promote(
        model_name="finetune_config_reranker",
        version="090",
        to_state="STAGING",
        actor="tester",
    )
    governance.promote(
        model_name="finetune_config_reranker",
        version="090",
        to_state="CANARY",
        actor="tester",
    )
    governance.promote(
        model_name="finetune_config_reranker",
        version="090",
        to_state="PRODUCTION",
        actor="tester",
    )

    model_state = governance.list_versions("finetune_config_reranker")
    assert model_state["production"] == "090"

    rollback = governance.rollback(
        model_name="finetune_config_reranker",
        actor="tester",
        reason="unit_test",
        to_version="100",
    )
    assert rollback["rolled_back_to"] == "100"
    assert governance.list_versions("finetune_config_reranker")["production"] == "100"
    assert len(store.list_governance_events(limit=50)) >= 1


def test_api_exposes_governance_and_retraining_endpoints(monkeypatch, tmp_path):
    api_module = _load_api_module(monkeypatch, tmp_path)
    client = TestClient(api_module.app)

    status_resp = client.get("/retraining/status")
    assert status_resp.status_code == 200
    assert "decision" in status_resp.json()

    promote_resp = client.post(
        "/governance/promote",
        json={
            "model_name": "finetune_config_reranker",
            "version": "777",
            "to_state": "CANDIDATE",
            "actor": "unit_test",
        },
    )
    assert promote_resp.status_code == 200
    assert promote_resp.json()["promotion"]["to_state"] == "CANDIDATE"

    events_resp = client.get("/governance/events", params={"limit": 20})
    assert events_resp.status_code == 200
    assert events_resp.json()["count"] >= 1
