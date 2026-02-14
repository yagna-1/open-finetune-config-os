from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ft_config_engine.memory import estimate_training_vram_gb_per_gpu
from ft_config_engine.models import RecommendationRequest
from ft_config_engine.normalization import bucket_model_size
from ft_config_engine.recommender import build_engine_from_dataset

from .gates import GateResult, evaluate_gates


GOLDEN_PATH = Path("evaluation/golden_dataset.jsonl")


@dataclass(slots=True)
class EvalReport:
    candidate_version: str
    sample_count: int
    overall_accuracy: float
    lr_log_mae: float
    oom_violation_rate: float
    ece: float
    overconfidence_rate: float
    per_category_acc: dict[str, float]
    css_integrity: bool
    gate_results: list[GateResult]
    all_gates_passed: bool
    per_example: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "gate_results": [gate.to_dict() for gate in self.gate_results],
        }


def run_evaluation(
    *,
    candidate_version: str,
    dataset_path: str | Path = "finetuning_configs_final.jsonl",
    golden_path: str | Path = GOLDEN_PATH,
    ml_reranker_path: str | Path | None = "artifacts/ml_reranker.joblib",
    hp_predictor_path: str | Path | None = "artifacts/hp_predictor.joblib",
) -> EvalReport:
    golden_records = _load_golden(golden_path)
    if not golden_records:
        raise ValueError("golden_dataset_empty")

    engine = build_engine_from_dataset(
        dataset_path,
        ml_reranker_path=Path(ml_reranker_path) if ml_reranker_path else None,
        hp_predictor_path=Path(hp_predictor_path) if hp_predictor_path else None,
    )

    results: list[dict[str, Any]] = []
    for example in golden_records:
        request_payload = dict(example.get("request") or {})
        gt = dict(example.get("ground_truth") or {})
        acceptable = dict(example.get("acceptable_ranges") or {})
        model_params = int(request_payload.get("model_params") or gt.get("model_params") or 1_000_000_000)
        gpu_vram_gb = float(request_payload.get("gpu_vram_gb") or 16.0)
        req = _to_request(request_payload)

        try:
            rec = engine.recommend(req, render_notebook=False)
        except Exception as exc:  # noqa: BLE001
            category = str(example.get("category") or "unknown")
            results.append(
                {
                    "id": str(example.get("id") or ""),
                    "category": category,
                    "difficulty": str(example.get("difficulty") or "unknown"),
                    "lr_ok": False,
                    "bs_ok": False,
                    "rank_ok": False,
                    "all_ok": False,
                    "lr_log_err": 1.0,
                    "would_oom": True,
                    "css_correct": False if category == "ood_synthetic" else True,
                    "confidence": 0.0,
                    "pred_learning_rate": 0.0,
                    "pred_batch_size": 0,
                    "pred_lora_rank": 0,
                    "pred_adapter_type": str(req.adapter_type),
                    "error": str(exc),
                }
            )
            continue
        hp = rec.safe_hyperparameters
        basis = rec.recommendation_basis

        pred_lr = float(hp.get("learning_rate") or 0.0)
        pred_bs = int(hp.get("batch_size_per_device") or 1)
        pred_rank = int(hp.get("lora_rank") or 0)
        gt_lr = float(gt.get("learning_rate") or max(pred_lr, 1e-9))

        lr_range = acceptable.get("learning_rate") or [pred_lr, pred_lr]
        bs_range = acceptable.get("batch_size") or [pred_bs, pred_bs]
        rank_range = acceptable.get("lora_r")

        lr_ok = float(lr_range[0]) <= pred_lr <= float(lr_range[1])
        bs_ok = int(bs_range[0]) <= pred_bs <= int(bs_range[1])
        rank_ok = True if rank_range is None else (int(rank_range[0]) <= pred_rank <= int(rank_range[1]))
        all_ok = bool(lr_ok and bs_ok and rank_ok)

        lr_log_err = abs(math.log10(max(pred_lr, 1e-12)) - math.log10(max(gt_lr, 1e-12)))
        vram_est = estimate_training_vram_gb_per_gpu(
            parameter_count=model_params,
            adapter_type=str(hp.get("adapter_type") or req.adapter_type),
            precision=str(hp.get("precision") or "fp16"),
            batch_size_per_device=pred_bs,
            sequence_length=int(hp.get("max_seq_length") or 512),
            num_gpus=int(hp.get("num_gpus") or 1),
            lora_rank=(int(hp.get("lora_rank")) if hp.get("lora_rank") is not None else None),
        )
        would_oom = bool(vram_est > (0.92 * gpu_vram_gb))

        css_correct = True
        if str(example.get("category") or "").strip().lower() == "ood_synthetic":
            css_correct = bool(basis.get("ood_guard_active"))

        results.append(
            {
                "id": str(example.get("id") or ""),
                "category": str(example.get("category") or "unknown"),
                "difficulty": str(example.get("difficulty") or "unknown"),
                "lr_ok": lr_ok,
                "bs_ok": bs_ok,
                "rank_ok": rank_ok,
                "all_ok": all_ok,
                "lr_log_err": float(lr_log_err),
                "would_oom": would_oom,
                "css_correct": css_correct,
                "confidence": float(basis.get("confidence_score") or 0.0),
                "pred_learning_rate": pred_lr,
                "pred_batch_size": pred_bs,
                "pred_lora_rank": pred_rank,
                "pred_adapter_type": str(hp.get("adapter_type") or ""),
            }
        )

    n = float(len(results))
    overall_acc = sum(1.0 for row in results if row["all_ok"]) / n
    lr_log_mae = sum(float(row["lr_log_err"]) for row in results) / n
    oom_viol_rate = sum(1.0 for row in results if row["would_oom"]) / n
    per_cat = _per_category_accuracy(results)
    ece = _compute_ece([(float(row["confidence"]), bool(row["all_ok"])) for row in results])
    overconf = sum(1.0 for row in results if row["confidence"] >= 0.70 and not row["all_ok"]) / n
    css_integrity = all(
        bool(row["css_correct"])
        for row in results
        if str(row["category"]).strip().lower() == "ood_synthetic"
    )

    prod_metrics = _load_production_metrics()
    gates = evaluate_gates(
        overall_acc=overall_acc,
        lr_mae=lr_log_mae,
        oom_rate=oom_viol_rate,
        ece=ece,
        overconf=overconf,
        per_cat=per_cat,
        css_integrity=css_integrity,
        current_prod_overall_acc=float(prod_metrics.get("overall_accuracy") or 0.0),
        current_prod_per_cat=dict(prod_metrics.get("per_category_acc") or {}),
    )

    return EvalReport(
        candidate_version=candidate_version,
        sample_count=int(n),
        overall_accuracy=float(round(overall_acc, 6)),
        lr_log_mae=float(round(lr_log_mae, 6)),
        oom_violation_rate=float(round(oom_viol_rate, 6)),
        ece=float(round(ece, 6)),
        overconfidence_rate=float(round(overconf, 6)),
        per_category_acc={k: float(round(v, 6)) for k, v in per_cat.items()},
        css_integrity=bool(css_integrity),
        gate_results=gates,
        all_gates_passed=all(gate.passed for gate in gates),
        per_example=results,
    )


def _load_golden(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    raw = Path(path).read_text(encoding="utf-8").splitlines()
    for line in raw:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _to_request(payload: dict[str, Any]) -> RecommendationRequest:
    model_params = payload.get("model_params")
    model_size_bucket = payload.get("model_size_bucket")
    if model_size_bucket is None and model_params:
        model_size_bucket = bucket_model_size(int(model_params))
    gpu_override = _map_gpu_override(
        gpu_type=str(payload.get("gpu_type") or "").strip(),
        gpu_vram_gb=float(payload.get("gpu_vram_gb") or 0.0),
    )
    return RecommendationRequest(
        platform=str(payload.get("platform") or "colab"),
        plan=str(payload.get("plan") or "free"),
        task_type=str(payload.get("task_type") or "instruction_following"),
        adapter_type=str(payload.get("adapter_type") or "qlora"),
        model_size_bucket=str(model_size_bucket or "medium"),
        model_name=payload.get("model_id"),
        model_parameter_count=(str(int(model_params)) if model_params is not None else None),
        dataset_name=payload.get("dataset_name"),
        dataset_size=(int(payload["dataset_size"]) if payload.get("dataset_size") is not None else None),
        sequence_length=(int(payload["sequence_length"]) if payload.get("sequence_length") is not None else None),
        num_gpus=(int(payload["num_gpus"]) if payload.get("num_gpus") is not None else None),
        gpu_override=gpu_override,
        epochs=(float(payload["epochs"]) if payload.get("epochs") is not None else None),
        strategy=str(payload.get("strategy") or "deterministic"),
        rerank_top_k=int(payload.get("rerank_top_k") or 5),
    )


def _map_gpu_override(*, gpu_type: str, gpu_vram_gb: float) -> str | None:
    canonical = gpu_type.strip().upper()
    if canonical == "T4":
        return "T4_16GB"
    if canonical == "P100":
        return "P100_16GB"
    if canonical == "V100":
        return "V100_16GB" if gpu_vram_gb <= 16.0 else None
    if canonical == "A100":
        return "A100_40GB" if gpu_vram_gb >= 40.0 else None
    if canonical == "A10G":
        return "A10G_24GB"
    return None


def _per_category_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[bool]] = {}
    for row in rows:
        category = str(row.get("category") or "unknown")
        buckets.setdefault(category, []).append(bool(row.get("all_ok")))
    return {
        category: (sum(1.0 for value in values if value) / max(1.0, float(len(values))))
        for category, values in buckets.items()
    }


def _compute_ece(pairs: list[tuple[float, bool]], n_bins: int = 10) -> float:
    if not pairs:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, correct in pairs:
        idx = min(int(max(0.0, min(conf, 1.0)) * n_bins), n_bins - 1)
        bins[idx].append((conf, correct))
    total = float(len(pairs))
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(conf for conf, _ in bucket) / len(bucket)
        avg_acc = sum(1.0 for _, ok in bucket if ok) / len(bucket)
        ece += abs(avg_conf - avg_acc) * (len(bucket) / total)
    return float(ece)


def _load_production_metrics() -> dict[str, Any]:
    baseline_path = Path("artifacts/eval_reports/production_metrics.json")
    if baseline_path.exists():
        try:
            return json.loads(baseline_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}
