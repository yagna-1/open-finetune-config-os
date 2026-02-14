from __future__ import annotations

from pathlib import Path

import pytest

from ft_config_engine.hp_predictor import (
    load_hyperparameter_predictor,
    save_hyperparameter_predictor,
    train_hyperparameter_predictor,
)
from ft_config_engine.recommender import build_engine_from_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "finetuning_configs_final.jsonl"


@pytest.fixture(scope="module")
def training_configs():
    dataset_arg = f"{DATASET_PATH},{DATASET_PATH}"
    engine = build_engine_from_dataset(
        dataset_arg,
        ml_reranker_path=None,
        hp_predictor_path=None,
    )
    return engine.configs[:500]


@pytest.fixture(scope="module")
def trained_predictor(training_configs):
    predictor, metadata = train_hyperparameter_predictor(training_configs, seed=11)
    return predictor, metadata


def test_train_hp_predictor_produces_target_metrics(trained_predictor):
    _, metadata = trained_predictor
    assert metadata.train_rows > 0
    assert metadata.validation_rows > 0
    assert metadata.dataset_configs > 0
    assert metadata.trained_targets
    assert "log10_learning_rate" in metadata.target_metrics
    assert "log2_effective_batch_size" in metadata.target_metrics
    assert "max_seq_length" in metadata.target_metrics


def test_hp_predictor_predict_returns_structured_outputs(trained_predictor):
    predictor, _ = trained_predictor
    prediction = predictor.predict(
        task_type="instruction_following",
        model_size_bucket="medium",
        adapter_type="qlora",
        model_architecture="LlamaForCausalLM",
        model_parameter_count=7_000_000_000,
        dataset_size=52_000,
        dataset_size_bucket="medium",
        gpu_memory_gb=16.0,
        num_gpus=1,
        requested_sequence_length=1024,
    )

    assert "log10_learning_rate" in prediction
    assert "log2_effective_batch_size" in prediction
    assert "max_seq_length" in prediction
    for payload in prediction.values():
        assert set(payload.keys()) == {"value", "low", "high", "confidence"}
        assert payload["low"] <= payload["value"] <= payload["high"]
        assert 0.0 <= payload["confidence"] <= 1.0


def test_hp_predictor_save_and_load_roundtrip(tmp_path, trained_predictor):
    predictor, metadata = trained_predictor
    path = tmp_path / "hp_predictor.joblib"
    saved = save_hyperparameter_predictor(predictor, path)
    loaded, status = load_hyperparameter_predictor(saved)

    assert status == "hp-predictor-loaded"
    assert loaded is not None
    assert loaded.model_version == metadata.model_version


def test_hp_predictor_load_rejects_invalid_payload(tmp_path):
    joblib = pytest.importorskip("joblib")
    artifact = tmp_path / "invalid_hp_predictor.joblib"
    joblib.dump({"metadata": {"model_version": "1.0.0"}, "models": {}}, artifact)

    loaded, status = load_hyperparameter_predictor(artifact)
    assert loaded is None
    assert status in {"hp-predictor-invalid-payload", "hp-predictor-empty-models"}
