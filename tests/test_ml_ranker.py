from __future__ import annotations

import math
from pathlib import Path

import pytest

from ft_config_engine.ml_ranker import (
    load_ml_reranker,
    save_ml_reranker,
    train_ml_reranker,
)
from ft_config_engine.models import RecommendationRequest
from ft_config_engine.recommender import build_engine_from_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "finetuning_configs_final.jsonl"


@pytest.fixture(scope="module")
def training_configs():
    # Passing a multi-token path disables auto-discovery of additional local JSONL files.
    dataset_arg = f"{DATASET_PATH},{DATASET_PATH}"
    engine = build_engine_from_dataset(dataset_arg, ml_reranker_path=None)
    return engine.configs[:400]


@pytest.fixture(scope="module")
def trained_model(training_configs):
    reranker, metadata = train_ml_reranker(training_configs, seed=17)
    return reranker, metadata


def test_train_ml_reranker_produces_metrics(trained_model):
    _, metadata = trained_model
    assert metadata.train_rows > 0
    assert metadata.validation_rows > 0
    assert metadata.train_queries > 0
    assert metadata.validation_queries > 0

    assert "rmse" in metadata.metrics
    assert "mae" in metadata.metrics
    assert "ndcg_at_5" in metadata.metrics
    assert "top1_regret" in metadata.metrics
    assert "oom_violation_rate" in metadata.metrics

    assert metadata.metrics["rmse"] >= 0.0
    assert metadata.metrics["mae"] >= 0.0
    assert 0.0 <= metadata.metrics["ndcg_at_5"] <= 1.0
    assert 0.0 <= metadata.metrics["top1_regret"] <= 1.0
    assert 0.0 <= metadata.metrics["oom_violation_rate"] <= 1.0


def test_save_and_load_ml_reranker_roundtrip(tmp_path, trained_model):
    reranker, metadata = trained_model
    model_path = tmp_path / "ml_reranker.joblib"

    saved_path = save_ml_reranker(reranker, model_path)
    loaded, status = load_ml_reranker(saved_path)

    assert status == "ml-reranker-loaded"
    assert loaded is not None
    assert loaded.model_version == metadata.model_version


def test_predict_scores_returns_all_candidates(training_configs, trained_model):
    reranker, _ = trained_model
    candidates = training_configs[:6]

    request = RecommendationRequest(
        platform="colab",
        plan="free",
        task_type="chatbot",
        adapter_type="qlora",
        model_size_bucket="medium",
        sequence_length=1024,
        num_gpus=1,
    )

    scores = reranker.predict_scores(
        request=request,
        task_type="chat",
        model_size_bucket="medium",
        adapter_type="qlora",
        candidates=candidates,
        gpu_memory_gb=16.0,
    )

    assert set(scores.keys()) == {candidate.record_id for candidate in candidates}
    assert all(math.isfinite(value) for value in scores.values())


def test_load_ml_reranker_rejects_invalid_payload(tmp_path):
    joblib = pytest.importorskip("joblib")
    artifact = tmp_path / "invalid_ml_reranker.joblib"
    joblib.dump({"metadata": {"model_version": "1.0.0"}}, artifact)

    loaded, status = load_ml_reranker(artifact)
    assert loaded is None
    assert status == "ml-reranker-invalid-payload"
