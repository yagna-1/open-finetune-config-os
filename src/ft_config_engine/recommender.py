from __future__ import annotations

import math
import re
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median

from .constants import (
    GPU_SPECS,
    NOTEBOOK_DATASET_ALIASES,
    NOTEBOOK_GATED_MODEL_PREFIXES,
    NOTEBOOK_MODEL_ALIASES,
    PLATFORM_GPU_MATRIX,
    RECOMMENDATION_MIN_SAMPLE,
)
from .dependencies import dependency_stack_for_platform
from .memory import (
    estimate_training_time_hours,
    estimate_training_vram_gb_per_gpu,
    max_safe_lora_rank,
    max_safe_seq_length,
)
from .hp_predictor import HyperparameterPredictor, load_hyperparameter_predictor
from .ml_ranker import MLReranker, load_ml_reranker
from .reranking import (
    aggregate_from_ranked_candidates,
    rank_candidates,
    rerank_with_ml_scores,
    ranked_candidates_summary,
)
from .models import (
    NormalizationReport,
    NormalizedConfig,
    RecommendationRequest,
    RecommendationResult,
    StatisticalProfile,
)
from .normalization import (
    bucket_model_size,
    canonicalize_task_type,
    load_and_prepare_datasets,
    parse_parameter_count,
)
from .notebook_engine import NotebookTemplateEngine
from .statistics import build_statistical_profiles


_TASK_TO_TEMPLATE = {
    "question_answering": "qa",
    "medical_qa": "qa",
    "financial_qa": "qa",
    "reading_comprehension": "qa",
    "summarization": "summarization_t5",
    "text_classification": "classification",
    "sentiment_analysis": "classification",
    "acceptability_classification": "classification",
    "question_classification": "classification",
    "paraphrase_detection": "classification",
    "named_entity_recognition": "classification",
    "legal_classification": "classification",
}

_MODEL_PARAM_NAME_RE = re.compile(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*([bm])(?![a-z0-9])", flags=re.IGNORECASE)


class ConfigRecommendationEngine:
    def __init__(
        self,
        configs: list[NormalizedConfig],
        profiles: dict[tuple[str, str, str], StatisticalProfile],
        normalization_report: NormalizationReport,
        template_engine: NotebookTemplateEngine | None = None,
        ml_reranker: MLReranker | None = None,
        ml_reranker_status: str = "ml-reranker-not-configured",
        hp_predictor: HyperparameterPredictor | None = None,
        hp_predictor_status: str = "hp-predictor-not-configured",
    ) -> None:
        self.configs = configs
        self.profiles = profiles
        self.normalization_report = normalization_report
        self.template_engine = template_engine
        self.ml_reranker = ml_reranker
        self.ml_reranker_status = ml_reranker_status
        self.hp_predictor = hp_predictor
        self.hp_predictor_status = hp_predictor_status
        self.model_parameter_count_by_name = self._build_model_parameter_count_index(configs)
        self.known_task_types = {cfg.task_type for cfg in configs}

    def available_gpus(self, platform: str, plan: str) -> list[str]:
        platform_key = self._platform_key(platform=platform, plan=plan)
        return PLATFORM_GPU_MATRIX[platform_key][:]

    def recommend(
        self,
        request: RecommendationRequest,
        render_notebook: bool = True,
        brain_strategy_hint: str | None = None,
        brain_decision: dict | None = None,
    ) -> RecommendationResult:
        notes: list[str] = []
        platform_key = self._platform_key(platform=request.platform, plan=request.plan)
        gpu_name = self._select_gpu(platform_key=platform_key, override=request.gpu_override, notes=notes)
        gpu_spec = GPU_SPECS[gpu_name]

        requested_task_type = request.task_type.strip().lower()
        task_type = canonicalize_task_type(requested_task_type)
        if task_type != requested_task_type:
            notes.append(f"task normalized from '{request.task_type}' to '{task_type}'")
        unsupported_task = task_type not in self.known_task_types
        if unsupported_task:
            notes.append(
                "requested task has no direct corpus support; applying OOD safety guard and generic fallback policy"
            )

        adapter_type = request.adapter_type.strip().lower()
        if adapter_type not in {"none", "lora", "qlora"}:
            raise ValueError(f"unsupported adapter_type: {request.adapter_type}")

        requested_adapter_type = adapter_type
        requested_model_size_bucket = self._resolve_model_size_bucket(request=request)
        model_size_bucket = requested_model_size_bucket
        pool, candidate_pool_match_type = self._select_candidate_pool(
            task_type=task_type,
            model_size_bucket=model_size_bucket,
            adapter_type=adapter_type,
            model_name=request.model_name,
        )
        if candidate_pool_match_type != "exact":
            notes.append(f"candidate pool fallback used: {candidate_pool_match_type}")

        if pool and all(cfg.model_size_bucket != requested_model_size_bucket for cfg in pool):
            model_size_bucket = self._mode_or_default(
                [cfg.model_size_bucket for cfg in pool],
                default=requested_model_size_bucket,
            )
            notes.append(
                f"model size adjusted from '{requested_model_size_bucket}' to '{model_size_bucket}' based on available evidence"
            )

        if pool and all(cfg.adapter_type != requested_adapter_type for cfg in pool):
            adapter_type = self._mode_or_default(
                [cfg.adapter_type for cfg in pool],
                default=requested_adapter_type,
            )
            notes.append(
                f"adapter adjusted from '{requested_adapter_type}' to '{adapter_type}' based on task evidence"
            )

        profile, profile_match_type = self._select_profile(
            task_type=task_type,
            model_size_bucket=model_size_bucket,
            adapter_type=adapter_type,
        )
        if profile_match_type != "exact":
            notes.append(f"profile fallback used: {profile_match_type}")

        resolved_num_gpus = self._resolve_num_gpus(
            request_num_gpus=request.num_gpus,
            selected_gpu=gpu_name,
            notes=notes,
        )
        prefilter_sequence_length = self._resolve_initial_sequence_length(
            requested_sequence_length=request.sequence_length,
            task_type=task_type,
            model_size_bucket=model_size_bucket,
            adapter_type=adapter_type,
            gpu_memory_gb=gpu_spec.memory_gb,
        )

        safe_pool = self._filter_gpu_safe_pool(
            configs=pool,
            gpu_memory_gb=gpu_spec.memory_gb,
            sequence_length=prefilter_sequence_length,
            num_gpus=resolved_num_gpus,
        )

        source_pool = safe_pool or pool
        if not source_pool and profile is None:
            raise ValueError("no data available to build recommendation")

        requested_strategy = (request.strategy or "auto").strip().lower()
        if requested_strategy not in {"auto", "deterministic", "hybrid", "hybrid_ml"}:
            raise ValueError(f"unsupported strategy: {request.strategy}")

        ml_reranker_loaded = self.ml_reranker is not None
        if requested_strategy == "auto":
            if brain_strategy_hint in {"deterministic", "hybrid", "hybrid_ml"}:
                strategy = brain_strategy_hint
                notes.append(f"strategy auto-selected by learning brain to '{strategy}'")
            else:
                strategy = self._auto_strategy(
                    has_candidates=bool(source_pool),
                    ml_reranker_loaded=ml_reranker_loaded,
                )
                notes.append(f"strategy auto-selected to '{strategy}'")
        else:
            strategy = requested_strategy
        if requested_strategy == "hybrid_ml" and not ml_reranker_loaded:
            strategy = "hybrid"
            notes.append(
                "hybrid_ml requested but ML reranker is unavailable; fallback to hybrid"
            )

        rerank_top_k = max(1, int(request.rerank_top_k or 5))
        ranking_diagnostics: list[dict] = []
        resolved_request = replace(
            request,
            sequence_length=prefilter_sequence_length,
            num_gpus=resolved_num_gpus,
            strategy=strategy,
        )

        baseline_from_strategy: dict[str, float | int | str | None] = {}
        if strategy in {"hybrid", "hybrid_ml"} and source_pool:
            ranked_candidates = rank_candidates(
                candidates=source_pool,
                request=resolved_request,
                gpu_memory_gb=gpu_spec.memory_gb,
                num_gpus=resolved_num_gpus,
                sequence_length=prefilter_sequence_length,
            )
            if strategy == "hybrid_ml" and self.ml_reranker is not None:
                ml_scores = self.ml_reranker.predict_scores(
                    request=resolved_request,
                    task_type=task_type,
                    model_size_bucket=model_size_bucket,
                    adapter_type=adapter_type,
                    candidates=source_pool,
                    gpu_memory_gb=gpu_spec.memory_gb,
                )
                ranked_candidates = rerank_with_ml_scores(
                    ranked=ranked_candidates,
                    ml_scores=ml_scores,
                    ml_weight=0.65,
                )
                notes.append(f"hybrid_ml reranker active: model={self.ml_reranker.model_version}")
            baseline_from_strategy = aggregate_from_ranked_candidates(
                ranked=ranked_candidates,
                top_k=rerank_top_k,
                fallback_pool=source_pool,
            )
            ranking_diagnostics = ranked_candidates_summary(ranked_candidates, top_n=min(rerank_top_k, 5))
            if len(ranked_candidates) < rerank_top_k:
                notes.append(f"{strategy} reranker used with only {len(ranked_candidates)} candidates")
        else:
            if strategy in {"hybrid", "hybrid_ml"}:
                notes.append(f"{strategy} reranker requested but no candidates available; fallback to deterministic")
            baseline_from_strategy = {
                "learning_rate": self._median_or_default(
                    [row.learning_rate for row in source_pool],
                    default=profile.median_learning_rate if profile else 2e-4,
                ),
                "effective_batch_size": int(
                    round(
                        self._median_or_default(
                            [float(row.effective_batch_size) for row in source_pool],
                            default=float(profile.median_effective_batch_size) if profile else 16.0,
                        )
                    )
                ),
                "max_seq_length": int(
                    round(
                        self._median_or_default(
                            [float(row.max_seq_length) for row in source_pool],
                            default=float(profile.median_seq_length) if profile else 1024.0,
                        )
                    )
                ),
                "optimizer": self._mode_or_default(
                    [row.optimizer for row in source_pool],
                    default=profile.typical_optimizer if profile else "adamw_torch",
                ),
                "scheduler": self._mode_or_default(
                    [row.scheduler for row in source_pool],
                    default="linear",
                ),
                "precision": self._mode_or_default(
                    [row.precision for row in source_pool],
                    default=profile.typical_precision if profile else "fp16",
                ),
                "lora_rank": int(
                    round(
                        self._median_or_default(
                            [float(row.lora_rank) for row in source_pool if row.lora_rank is not None],
                            default=float(profile.median_lora_rank)
                            if profile and profile.median_lora_rank
                            else 16.0,
                        )
                    )
                )
                if adapter_type in {"lora", "qlora"}
                else None,
                "model_name": self._mode_or_default(
                    [row.model_name for row in source_pool],
                    default=self._default_model_name_for_bucket(model_size_bucket),
                ),
                "dataset_name": self._mode_or_default(
                    [row.dataset_name for row in source_pool],
                    default="custom_dataset",
                ),
            }

        model_name = request.model_name or str(
            baseline_from_strategy.get("model_name")
            or self._mode_or_default(
                [row.model_name for row in source_pool],
                default=self._default_model_name_for_bucket(model_size_bucket),
            )
        )

        parameter_count = self._resolve_parameter_count(
            request=request,
            source_pool=source_pool,
            model_size_bucket=model_size_bucket,
            model_name=model_name,
        )

        dataset_name = request.dataset_name or str(
            baseline_from_strategy.get("dataset_name")
            or self._mode_or_default(
                [row.dataset_name for row in source_pool],
                default="custom_dataset",
            )
        )
        dataset_rows = self._resolve_dataset_size(request=request, source_pool=source_pool)
        dataset_size_bucket = self._mode_or_default(
            [row.dataset_size_bucket for row in source_pool],
            default="medium",
        )
        model_architecture = self._mode_or_default(
            [row.model_architecture for row in source_pool if row.model_name == model_name],
            default=self._mode_or_default([row.model_architecture for row in source_pool], default="unknown"),
        )

        baseline_lr = float(
            baseline_from_strategy.get("learning_rate")
            or (
                profile.median_learning_rate
                if profile
                else self._median_or_default([row.learning_rate for row in source_pool], default=2e-4)
            )
        )
        baseline_effective_batch = int(
            round(
                float(
                    baseline_from_strategy.get("effective_batch_size")
                    or (
                        profile.median_effective_batch_size
                        if profile
                        else self._median_or_default(
                            [float(row.effective_batch_size) for row in source_pool],
                            default=16.0,
                        )
                    )
                )
            )
        )
        baseline_seq_length = int(
            round(
                float(
                    baseline_from_strategy.get("max_seq_length")
                    or (
                        profile.median_seq_length
                        if profile
                        else self._median_or_default(
                            [float(row.max_seq_length) for row in source_pool],
                            default=1024.0,
                        )
                    )
                )
            )
        )

        hp_predictor_loaded = self.hp_predictor is not None
        hp_prediction_payload: dict[str, dict[str, float]] = {}
        if self.hp_predictor is not None:
            try:
                hp_prediction_payload = self.hp_predictor.predict(
                    task_type=task_type,
                    model_size_bucket=model_size_bucket,
                    adapter_type=adapter_type,
                    model_architecture=model_architecture,
                    model_parameter_count=parameter_count,
                    dataset_size=dataset_rows,
                    dataset_size_bucket=dataset_size_bucket,
                    gpu_memory_gb=gpu_spec.memory_gb,
                    num_gpus=resolved_num_gpus,
                    requested_sequence_length=prefilter_sequence_length,
                )
            except Exception:  # noqa: BLE001
                hp_prediction_payload = {}

        if hp_prediction_payload:
            lr_pred = hp_prediction_payload.get("log10_learning_rate")
            if lr_pred:
                baseline_lr = self._blend_numeric(
                    baseline=baseline_lr,
                    predicted=float(lr_pred["value"]),
                    confidence=float(lr_pred.get("confidence") or 0.0),
                    max_weight=0.45,
                )

            batch_pred = hp_prediction_payload.get("log2_effective_batch_size")
            if batch_pred:
                predicted_batch = max(1.0, float(batch_pred["value"]))
                blended_batch = self._blend_numeric(
                    baseline=float(baseline_effective_batch),
                    predicted=predicted_batch,
                    confidence=float(batch_pred.get("confidence") or 0.0),
                    max_weight=0.45,
                )
                baseline_effective_batch = max(1, int(round(blended_batch)))

            seq_pred = hp_prediction_payload.get("max_seq_length")
            if seq_pred:
                predicted_seq = max(128.0, float(seq_pred["value"]))
                blended_seq = self._blend_numeric(
                    baseline=float(baseline_seq_length),
                    predicted=predicted_seq,
                    confidence=float(seq_pred.get("confidence") or 0.0),
                    max_weight=0.35,
                )
                baseline_seq_length = max(128, int(round(blended_seq / 64.0) * 64))

            avg_conf = sum(
                float(payload.get("confidence") or 0.0) for payload in hp_prediction_payload.values()
            ) / max(1, len(hp_prediction_payload))
            notes.append(f"ml hyperparameter predictor applied (avg confidence={avg_conf:.2f})")

        fallback_guard = profile_match_type != "exact" or len(pool) == 0
        strict_ood_guard = unsupported_task or len(pool) == 0
        if fallback_guard and gpu_name != "CPU":
            fallback_batch_cap = self._fallback_effective_batch_cap(
                gpu_memory_gb=gpu_spec.memory_gb,
                model_size_bucket=model_size_bucket,
                adapter_type=adapter_type,
            )
            if baseline_effective_batch > fallback_batch_cap:
                baseline_effective_batch = fallback_batch_cap
                notes.append(
                    f"fallback confidence guard applied: effective batch capped to {fallback_batch_cap}"
                )
        if strict_ood_guard and baseline_effective_batch > 4:
            baseline_effective_batch = 4
            notes.append("OOD guard applied: effective batch capped to 4")
        baseline_optimizer = str(
            baseline_from_strategy.get("optimizer")
            or self._mode_or_default(
                [row.optimizer for row in source_pool],
                default=profile.typical_optimizer if profile else "adamw_torch",
            )
        )
        baseline_scheduler = str(
            baseline_from_strategy.get("scheduler")
            or self._mode_or_default(
                [row.scheduler for row in source_pool],
                default="linear",
            )
        )
        baseline_precision = str(
            baseline_from_strategy.get("precision")
            or self._mode_or_default(
                [row.precision for row in source_pool],
                default=profile.typical_precision if profile else "fp16",
            )
        )

        lr_q1 = profile.learning_rate_q1 if profile else baseline_lr * 0.8
        lr_q3 = profile.learning_rate_q3 if profile else baseline_lr * 1.2
        learning_rate = max(lr_q1, min(baseline_lr, lr_q3))

        if gpu_name == "CPU":
            precision = "fp32"
            notes.append("CPU selected: precision forced to fp32 and memory checks skipped")
        else:
            precision = baseline_precision
            if precision == "bf16" and not gpu_spec.supports_bf16:
                precision = "fp16"
                notes.append("bf16 not supported on selected GPU, fell back to fp16")

        sequence_length = max(128, request.sequence_length or baseline_seq_length)
        if gpu_name != "CPU":
            gpu_seq_cap = max_safe_seq_length(gpu_spec.memory_gb)
            if sequence_length > gpu_seq_cap:
                sequence_length = gpu_seq_cap
                notes.append(f"sequence length clamped to {gpu_seq_cap} for GPU safety")
            if fallback_guard:
                conservative_seq_cap = self._fallback_seq_cap(
                    gpu_memory_gb=gpu_spec.memory_gb,
                    model_size_bucket=model_size_bucket,
                )
                if sequence_length > conservative_seq_cap:
                    sequence_length = conservative_seq_cap
                    notes.append(
                        f"fallback confidence guard applied: sequence length capped to {conservative_seq_cap}"
                    )
            if strict_ood_guard and sequence_length > 512:
                sequence_length = 512
                notes.append("OOD guard applied: sequence length capped to 512")
        if request.sequence_length is None:
            notes.append(f"sequence length auto-set to {sequence_length}")

        lora_rank: int | None = None
        if adapter_type in {"lora", "qlora"}:
            strategy_rank = baseline_from_strategy.get("lora_rank")
            if strategy_rank is not None:
                lora_rank = int(strategy_rank)
            else:
                baseline_rank = self._median_or_default(
                    [float(row.lora_rank) for row in source_pool if row.lora_rank is not None],
                    default=float(profile.median_lora_rank) if profile and profile.median_lora_rank else 16.0,
                )
                lora_rank = int(round(baseline_rank))
            rank_pred = hp_prediction_payload.get("lora_rank")
            if rank_pred:
                blended_rank = self._blend_numeric(
                    baseline=float(max(4, lora_rank)),
                    predicted=max(4.0, float(rank_pred["value"])),
                    confidence=float(rank_pred.get("confidence") or 0.0),
                    max_weight=0.35,
                )
                lora_rank = int(round(blended_rank))
            lora_rank = max(4, lora_rank)
            if gpu_name != "CPU":
                rank_cap = max_safe_lora_rank(gpu_spec.memory_gb)
                if lora_rank > rank_cap:
                    lora_rank = rank_cap
                    notes.append(f"LoRA rank clamped to {rank_cap} for GPU safety")

        shape = self._choose_safe_training_shape(
            parameter_count=parameter_count,
            adapter_type=adapter_type,
            precision=precision,
            initial_sequence_length=sequence_length,
            target_effective_batch=max(1, baseline_effective_batch),
            num_gpus=resolved_num_gpus,
            gpu_memory_gb=gpu_spec.memory_gb,
            lora_rank=lora_rank,
            notes=notes,
        )
        if strict_ood_guard and int(shape["batch_size_per_device"]) > 4:
            constrained_batch = 4
            constrained_grad = max(1, math.ceil(max(1, baseline_effective_batch) / (constrained_batch * resolved_num_gpus)))
            constrained_effective = constrained_batch * constrained_grad * resolved_num_gpus
            constrained_est = estimate_training_vram_gb_per_gpu(
                parameter_count=parameter_count,
                adapter_type=adapter_type,
                precision=precision,
                batch_size_per_device=constrained_batch,
                sequence_length=int(shape["sequence_length"]),
                num_gpus=resolved_num_gpus,
                lora_rank=lora_rank,
            )
            shape = {
                **shape,
                "batch_size_per_device": constrained_batch,
                "gradient_accumulation_steps": constrained_grad,
                "effective_batch_size": constrained_effective,
                "estimated_vram_gb_per_gpu": constrained_est,
            }
            notes.append("OOD guard applied: per-device batch constrained to 4")

        if adapter_type == "qlora":
            baseline_optimizer = "paged_adamw_32bit"
        resolved_epochs = self._resolve_epochs(
            requested_epochs=request.epochs,
            dataset_rows=dataset_rows,
            source_pool=source_pool,
            task_type=task_type,
            adapter_type=adapter_type,
            notes=notes,
        )
        est_hours = estimate_training_time_hours(
            dataset_rows=dataset_rows,
            sequence_length=shape["sequence_length"],
            epochs=resolved_epochs,
            effective_batch_size=shape["effective_batch_size"],
            selected_gpu=gpu_name,
            num_gpus=resolved_num_gpus,
            adapter_type=adapter_type,
        )

        dependencies = dependency_stack_for_platform(request.platform)
        confidence_score, confidence_level = self._recommendation_confidence(
            request=request,
            profile_match_type=profile_match_type,
            candidate_pool_size=len(pool),
            gpu_safe_pool_size=len(safe_pool),
            profile_sample_size=(profile.sample_size if profile else 0),
            unsupported_task=unsupported_task,
        )
        if confidence_level == "low":
            notes.append("low confidence recommendation: use smaller batches and verify with a short pilot run")

        template_name = self._select_template(task_type=task_type, adapter_type=adapter_type)
        template_fallback_warning: str | None = None
        if unsupported_task:
            template_fallback_warning = (
                f"no task-specific template for '{task_type}'; using adapter-driven fallback template '{template_name}'"
            )
            notes.append(template_fallback_warning)
        notebook_json = None
        if render_notebook and self.template_engine is not None:
            notebook_json = self.template_engine.render_template(
                template_name=template_name,
                context={
                    "platform": request.platform,
                    "plan": request.plan,
                    "gpu": gpu_name,
                    "dependencies": dependencies,
                    "task_type": task_type,
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "learning_rate": learning_rate,
                    "epochs": resolved_epochs,
                    "batch_size_per_device": shape["batch_size_per_device"],
                    "gradient_accumulation_steps": shape["gradient_accumulation_steps"],
                    "effective_batch_size": shape["effective_batch_size"],
                    "max_seq_length": shape["sequence_length"],
                    "precision": precision,
                    "optimizer": baseline_optimizer,
                    "scheduler": baseline_scheduler,
                    "adapter_type": adapter_type,
                    "lora_rank": lora_rank or 0,
                    "push_to_hub": request.push_to_hub,
                    "huggingface_repo_id": request.huggingface_repo_id or "",
                    "dataset_aliases": NOTEBOOK_DATASET_ALIASES,
                    "model_aliases": NOTEBOOK_MODEL_ALIASES,
                    "gated_model_prefixes": NOTEBOOK_GATED_MODEL_PREFIXES,
                },
            )

        safe_hparams = {
            "learning_rate": learning_rate,
            "epochs": resolved_epochs,
            "batch_size_per_device": shape["batch_size_per_device"],
            "gradient_accumulation_steps": shape["gradient_accumulation_steps"],
            "effective_batch_size": shape["effective_batch_size"],
            "max_seq_length": shape["sequence_length"],
            "optimizer": baseline_optimizer,
            "scheduler": baseline_scheduler,
            "precision": precision,
            "lora_rank": lora_rank,
            "adapter_type": adapter_type,
            "num_gpus": resolved_num_gpus,
            "model_name": model_name,
            "dataset_name": dataset_name,
        }

        basis = {
            "strategy": strategy,
            "requested_strategy": requested_strategy,
            "rerank_top_k": rerank_top_k,
            "requested_sequence_length": request.sequence_length,
            "requested_num_gpus": request.num_gpus,
            "requested_epochs": request.epochs,
            "requested_task_type": requested_task_type,
            "resolved_task_type": task_type,
            "requested_model_size_bucket": requested_model_size_bucket,
            "model_size_bucket": model_size_bucket,
            "requested_adapter_type": requested_adapter_type,
            "resolved_adapter_type": adapter_type,
            "candidate_pool_match_type": candidate_pool_match_type,
            "resolved_parameter_count": parameter_count,
            "resolved_sequence_length": shape["sequence_length"],
            "resolved_num_gpus": resolved_num_gpus,
            "resolved_epochs": resolved_epochs,
            "profile_match_type": profile_match_type,
            "profile_sample_size": profile.sample_size if profile else 0,
            "candidate_pool_size": len(pool),
            "gpu_safe_pool_size": len(safe_pool),
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
            "unsupported_task": unsupported_task,
            "ood_guard_active": strict_ood_guard,
            "template_fallback_warning": template_fallback_warning,
            "ranked_candidates": ranking_diagnostics,
            "normalization_report": asdict(self.normalization_report),
            "brain_strategy_hint": brain_strategy_hint,
            "brain_decision": brain_decision or {},
            "ml_reranker": {
                "loaded": ml_reranker_loaded,
                "status": self.ml_reranker_status,
                "model_version": self.ml_reranker.model_version if self.ml_reranker is not None else None,
            },
            "hp_predictor": {
                "loaded": hp_predictor_loaded,
                "status": self.hp_predictor_status,
                "model_version": self.hp_predictor.model_version if self.hp_predictor is not None else None,
                "prediction": hp_prediction_payload,
            },
        }

        return RecommendationResult(
            platform_key=platform_key,
            selected_gpu=gpu_name,
            selected_gpu_memory_gb=gpu_spec.memory_gb,
            safe_hyperparameters=safe_hparams,
            dependency_stack=dependencies,
            estimated_vram_gb_per_gpu=shape["estimated_vram_gb_per_gpu"],
            estimated_training_time_hours=est_hours,
            recommendation_basis=basis,
            notebook_template=template_name,
            notebook_json=notebook_json,
            notes=notes,
        )

    def _platform_key(self, platform: str, plan: str) -> str:
        key = f"{platform.strip().lower()}_{plan.strip().lower()}"
        if key not in PLATFORM_GPU_MATRIX:
            raise ValueError(f"unsupported platform/plan combination: {key}")
        return key

    def _select_gpu(self, platform_key: str, override: str | None, notes: list[str]) -> str:
        options = PLATFORM_GPU_MATRIX[platform_key]
        if override:
            cleaned = override.strip()
            if cleaned in options:
                return cleaned
            notes.append(f"gpu_override '{override}' unavailable for {platform_key}; default used")
        return options[0]

    def _resolve_model_size_bucket(self, request: RecommendationRequest) -> str:
        if request.model_size_bucket:
            return request.model_size_bucket.strip().lower()
        if request.model_parameter_count:
            return bucket_model_size(parse_parameter_count(request.model_parameter_count))
        return "medium"

    def _select_candidate_pool(
        self,
        task_type: str,
        model_size_bucket: str,
        adapter_type: str,
        model_name: str | None,
    ) -> tuple[list[NormalizedConfig], str]:
        fallback_steps = [
            (
                "exact",
                lambda cfg: cfg.task_type == task_type
                and cfg.model_size_bucket == model_size_bucket
                and cfg.adapter_type == adapter_type,
            ),
            (
                "task+model_size",
                lambda cfg: cfg.task_type == task_type and cfg.model_size_bucket == model_size_bucket,
            ),
            (
                "task+adapter",
                lambda cfg: cfg.task_type == task_type and cfg.adapter_type == adapter_type,
            ),
            (
                "task_only",
                lambda cfg: cfg.task_type == task_type,
            ),
        ]

        for label, predicate in fallback_steps:
            pool = [cfg for cfg in self.configs if predicate(cfg)]
            if not pool:
                continue
            if model_name:
                narrowed = [cfg for cfg in pool if cfg.model_name == model_name]
                if narrowed:
                    return narrowed, label
            return pool, label

        return [], "none"

    def _filter_gpu_safe_pool(
        self,
        configs: list[NormalizedConfig],
        gpu_memory_gb: float,
        sequence_length: int,
        num_gpus: int,
    ) -> list[NormalizedConfig]:
        if gpu_memory_gb <= 0:
            return configs[:]
        safe_pool: list[NormalizedConfig] = []
        for cfg in configs:
            est = estimate_training_vram_gb_per_gpu(
                parameter_count=cfg.model_parameter_count_num,
                adapter_type=cfg.adapter_type,
                precision=cfg.precision,
                batch_size_per_device=cfg.batch_size_per_device,
                sequence_length=min(sequence_length, cfg.max_seq_length),
                num_gpus=max(1, num_gpus),
                lora_rank=cfg.lora_rank,
            )
            if est <= gpu_memory_gb:
                safe_pool.append(cfg)
        return safe_pool

    def _resolve_parameter_count(
        self,
        request: RecommendationRequest,
        source_pool: list[NormalizedConfig],
        model_size_bucket: str,
        model_name: str | None,
    ) -> int:
        if request.model_parameter_count:
            return parse_parameter_count(request.model_parameter_count)

        if request.model_name and request.model_name in self.model_parameter_count_by_name:
            return self.model_parameter_count_by_name[request.model_name]

        if model_name and model_name in self.model_parameter_count_by_name:
            return self.model_parameter_count_by_name[model_name]

        inferred_from_name = self._infer_parameter_count_from_model_name(model_name)
        if inferred_from_name is not None:
            return inferred_from_name

        if source_pool:
            return int(median([cfg.model_parameter_count_num for cfg in source_pool]))

        defaults = {
            "small": 1_000_000_000,
            "medium": 7_000_000_000,
            "large": 13_000_000_000,
        }
        return defaults.get(model_size_bucket, 7_000_000_000)

    def _resolve_dataset_size(
        self,
        request: RecommendationRequest,
        source_pool: list[NormalizedConfig],
    ) -> int:
        if request.dataset_size and request.dataset_size > 0:
            return request.dataset_size

        if request.dataset_name:
            matches = [cfg.dataset_size for cfg in source_pool if cfg.dataset_name == request.dataset_name]
            if matches:
                return int(median(matches))

        if source_pool:
            return int(median([cfg.dataset_size for cfg in source_pool]))
        return 50_000

    def _select_profile(
        self,
        task_type: str,
        model_size_bucket: str,
        adapter_type: str,
    ) -> tuple[StatisticalProfile | None, str]:
        exact_key = (task_type, model_size_bucket, adapter_type)
        if exact_key in self.profiles:
            return self.profiles[exact_key], "exact"

        fallback_order = [
            (
                "task+model_size",
                [
                    profile
                    for profile in self.profiles.values()
                    if profile.task_type == task_type
                    and profile.model_size_bucket == model_size_bucket
                ],
            ),
            (
                "task+adapter",
                [
                    profile
                    for profile in self.profiles.values()
                    if profile.task_type == task_type and profile.adapter_type == adapter_type
                ],
            ),
            (
                "task_only",
                [profile for profile in self.profiles.values() if profile.task_type == task_type],
            ),
            (
                "adapter+model_size",
                [
                    profile
                    for profile in self.profiles.values()
                    if profile.adapter_type == adapter_type
                    and profile.model_size_bucket == model_size_bucket
                ],
            ),
            (
                "global",
                list(self.profiles.values()),
            ),
        ]

        for match_type, candidates in fallback_order:
            if not candidates:
                continue
            chosen = sorted(candidates, key=lambda p: p.sample_size, reverse=True)[0]
            return chosen, match_type

        return None, "none"

    def _choose_safe_training_shape(
        self,
        parameter_count: int,
        adapter_type: str,
        precision: str,
        initial_sequence_length: int,
        target_effective_batch: int,
        num_gpus: int,
        gpu_memory_gb: float,
        lora_rank: int | None,
        notes: list[str],
    ) -> dict[str, float | int]:
        target = max(1, target_effective_batch)
        seq_length = max(128, initial_sequence_length)
        rank = lora_rank
        min_seq = 128

        if gpu_memory_gb <= 0:
            batch_size_per_device = 1
            gradient_accumulation = max(1, target // max(1, num_gpus))
            effective_batch = batch_size_per_device * gradient_accumulation * max(1, num_gpus)
            return {
                "batch_size_per_device": batch_size_per_device,
                "gradient_accumulation_steps": gradient_accumulation,
                "effective_batch_size": effective_batch,
                "sequence_length": seq_length,
                "lora_rank": rank or 0,
                "estimated_vram_gb_per_gpu": 0.0,
            }

        for _ in range(36):
            best = self._find_best_batch_shape_for_limits(
                parameter_count=parameter_count,
                adapter_type=adapter_type,
                precision=precision,
                sequence_length=seq_length,
                target_effective_batch=target,
                num_gpus=num_gpus,
                gpu_memory_gb=gpu_memory_gb,
                lora_rank=rank,
            )
            if best is not None:
                if seq_length != initial_sequence_length:
                    notes.append(f"sequence length reduced to {seq_length} for OOM safety")
                if rank != lora_rank and adapter_type in {"lora", "qlora"}:
                    notes.append(f"LoRA rank reduced to {rank} for OOM safety")
                if target != target_effective_batch:
                    notes.append(f"effective batch reduced to {target} for OOM safety")
                return best

            if adapter_type in {"lora", "qlora"} and rank is not None and rank > 4:
                rank = max(4, rank // 2)
                continue

            if seq_length > min_seq:
                seq_length = max(min_seq, (seq_length * 3) // 4)
                seq_length = max(min_seq, (seq_length // 64) * 64)
                continue

            if target > 1:
                target = max(1, target // 2)
                continue

            break

        raise ValueError("unable to build a GPU-safe training configuration for this request")

    def _find_best_batch_shape_for_limits(
        self,
        parameter_count: int,
        adapter_type: str,
        precision: str,
        sequence_length: int,
        target_effective_batch: int,
        num_gpus: int,
        gpu_memory_gb: float,
        lora_rank: int | None,
    ) -> dict[str, float | int] | None:
        batch_candidates = [32, 16, 8, 4, 2, 1]
        gpu_count = max(1, num_gpus)
        target = max(1, target_effective_batch)

        for batch_size_per_device in batch_candidates:
            gradient_accumulation = max(1, math.ceil(target / (batch_size_per_device * gpu_count)))
            effective_batch = batch_size_per_device * gradient_accumulation * gpu_count
            est = estimate_training_vram_gb_per_gpu(
                parameter_count=parameter_count,
                adapter_type=adapter_type,
                precision=precision,
                batch_size_per_device=batch_size_per_device,
                sequence_length=sequence_length,
                num_gpus=gpu_count,
                lora_rank=lora_rank,
            )
            if est <= gpu_memory_gb:
                return {
                    "batch_size_per_device": batch_size_per_device,
                    "gradient_accumulation_steps": gradient_accumulation,
                    "effective_batch_size": effective_batch,
                    "sequence_length": sequence_length,
                    "lora_rank": lora_rank or 0,
                    "estimated_vram_gb_per_gpu": est,
                }

        return None

    def _select_template(self, task_type: str, adapter_type: str) -> str:
        if adapter_type == "qlora":
            return "qlora_4bit"
        if task_type in _TASK_TO_TEMPLATE:
            return _TASK_TO_TEMPLATE[task_type]
        if adapter_type == "lora":
            return "causal_lm_lora"
        return "classification"

    @staticmethod
    def _median_or_default(values: list[float], default: float) -> float:
        if not values:
            return default
        return float(median(values))

    @staticmethod
    def _blend_numeric(
        baseline: float,
        predicted: float,
        confidence: float,
        *,
        max_weight: float,
    ) -> float:
        confidence_clamped = max(0.0, min(1.0, confidence))
        weight = max(0.0, min(max_weight, max_weight * confidence_clamped))
        return ((1.0 - weight) * baseline) + (weight * predicted)

    @staticmethod
    def _mode_or_default(values: list[str], default: str) -> str:
        if not values:
            return default
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        max_count = max(counts.values())
        winners = sorted([value for value, count in counts.items() if count == max_count])
        return winners[0]

    @staticmethod
    def _build_model_parameter_count_index(configs: list[NormalizedConfig]) -> dict[str, int]:
        grouped: dict[str, list[int]] = {}
        for cfg in configs:
            grouped.setdefault(cfg.model_name, []).append(cfg.model_parameter_count_num)
        return {model_name: int(median(values)) for model_name, values in grouped.items()}

    @staticmethod
    def _infer_parameter_count_from_model_name(model_name: str | None) -> int | None:
        if not model_name:
            return None
        match = _MODEL_PARAM_NAME_RE.search(model_name.lower())
        if not match:
            return None
        value = float(match.group(1))
        unit = match.group(2).lower()
        multiplier = 1_000_000_000 if unit == "b" else 1_000_000
        return int(value * multiplier)

    @staticmethod
    def _auto_strategy(has_candidates: bool, ml_reranker_loaded: bool) -> str:
        if not has_candidates:
            return "deterministic"
        if ml_reranker_loaded:
            return "hybrid_ml"
        return "hybrid"

    @staticmethod
    def _resolve_num_gpus(
        request_num_gpus: int | None,
        selected_gpu: str,
        notes: list[str],
    ) -> int:
        if request_num_gpus is not None:
            resolved = max(1, int(request_num_gpus))
            if resolved > 1:
                notes.append("num_gpus override enabled; ensure multi-GPU runtime availability")
            return resolved

        if selected_gpu == "CPU":
            notes.append("num_gpus auto-set to 1 (CPU mode)")
            return 1

        notes.append("num_gpus auto-set to 1 (single accelerator assumption)")
        return 1

    @staticmethod
    def _resolve_initial_sequence_length(
        requested_sequence_length: int | None,
        task_type: str,
        model_size_bucket: str,
        adapter_type: str,
        gpu_memory_gb: float,
    ) -> int:
        if requested_sequence_length is not None:
            return max(128, int(requested_sequence_length))

        if task_type in {"instruction_following", "chat", "code_generation", "causal_lm"}:
            base = 1024
        elif task_type in {"summarization", "translation"}:
            base = 1024
        elif task_type in {"question_answering", "reading_comprehension"}:
            base = 768
        else:
            base = 512

        if adapter_type == "none" and model_size_bucket in {"medium", "large"} and 0 < gpu_memory_gb <= 16:
            base = min(base, 768)
        if adapter_type == "none" and model_size_bucket == "large" and 0 < gpu_memory_gb <= 24:
            base = min(base, 512)

        if gpu_memory_gb > 0:
            base = min(base, max_safe_seq_length(gpu_memory_gb))
        return max(128, int(base))

    @staticmethod
    def _resolve_epochs(
        requested_epochs: float | None,
        dataset_rows: int,
        source_pool: list[NormalizedConfig],
        task_type: str,
        adapter_type: str,
        notes: list[str],
    ) -> float:
        if requested_epochs is not None:
            return max(0.1, round(float(requested_epochs), 3))

        priors = [float(cfg.epochs) for cfg in source_pool if cfg.epochs > 0]
        prior_epochs = float(median(priors)) if priors else 3.0

        if dataset_rows >= 1_000_000:
            resolved = 1.0
        elif dataset_rows >= 250_000:
            resolved = min(prior_epochs, 2.0)
        elif dataset_rows >= 50_000:
            resolved = min(prior_epochs, 3.0)
        elif dataset_rows <= 5_000:
            resolved = max(prior_epochs, 4.0)
        else:
            resolved = prior_epochs

        if task_type in {"instruction_following", "chat", "code_generation"} and adapter_type in {"lora", "qlora"}:
            resolved = min(resolved, 3.0)

        resolved = max(1.0, min(6.0, round(float(resolved), 1)))
        notes.append(f"epochs auto-set to {resolved} based on dataset size and corpus priors")
        return resolved

    @staticmethod
    def _fallback_effective_batch_cap(
        gpu_memory_gb: float,
        model_size_bucket: str,
        adapter_type: str,
    ) -> int:
        if gpu_memory_gb <= 0:
            return 2
        if adapter_type == "none":
            return 4 if gpu_memory_gb <= 16 else 8
        if gpu_memory_gb <= 16:
            if model_size_bucket in {"medium", "large"}:
                return 8
            return 16
        if gpu_memory_gb <= 24:
            return 16
        return 32

    @staticmethod
    def _fallback_seq_cap(gpu_memory_gb: float, model_size_bucket: str) -> int:
        if gpu_memory_gb <= 0:
            return 512
        if gpu_memory_gb <= 16 and model_size_bucket in {"medium", "large"}:
            return 768
        if gpu_memory_gb <= 24 and model_size_bucket == "large":
            return 1024
        return max_safe_seq_length(gpu_memory_gb)

    @staticmethod
    def _recommendation_confidence(
        request: RecommendationRequest,
        profile_match_type: str,
        candidate_pool_size: int,
        gpu_safe_pool_size: int,
        profile_sample_size: int,
        unsupported_task: bool,
    ) -> tuple[float, str]:
        score = 0.92
        fallback_penalty = {
            "exact": 0.0,
            "task+model_size": 0.12,
            "task+adapter": 0.18,
            "task_only": 0.28,
            "adapter+model_size": 0.36,
            "global": 0.5,
            "none": 0.65,
        }
        score -= fallback_penalty.get(profile_match_type, 0.5)
        if unsupported_task:
            score -= 0.2
        if candidate_pool_size <= 0:
            score -= 0.2
        else:
            if candidate_pool_size < 20:
                sparsity_penalty = (20 - float(candidate_pool_size)) / 20.0
                score -= 0.2 * max(0.0, min(1.0, sparsity_penalty))
            safe_ratio = float(gpu_safe_pool_size) / float(max(1, candidate_pool_size))
            if safe_ratio < 0.8:
                score -= (0.8 - safe_ratio) * 0.25
        if profile_sample_size < 25:
            sample_penalty = (25 - float(max(0, profile_sample_size))) / 25.0
            score -= 0.15 * max(0.0, min(1.0, sample_penalty))
        if not request.model_name:
            score -= 0.07
        if not request.dataset_name:
            score -= 0.07
        if (
            not unsupported_task
            and profile_match_type == "exact"
            and request.model_name
            and request.dataset_name
            and candidate_pool_size >= 30
            and gpu_safe_pool_size >= 10
        ):
            if candidate_pool_size >= 200:
                score += 0.24
            elif candidate_pool_size >= 100:
                score += 0.2
            else:
                score += 0.16

        if not unsupported_task:
            bounded = max(0.0, min(1.0, score))
            # Calibrate in-distribution confidence upward so confidence aligns with high-coverage profile accuracy.
            score = 1.0 - ((1.0 - bounded) ** 2)

        score = max(0.0, min(1.0, round(score, 3)))
        if score >= 0.8:
            return score, "high"
        if score >= 0.6:
            return score, "moderate"
        return score, "low"

    @staticmethod
    def _default_model_name_for_bucket(model_size_bucket: str) -> str:
        defaults = {
            "small": "distilbert-base-uncased",
            "medium": "meta-llama/Llama-2-7b-hf",
            "large": "meta-llama/Llama-2-13b-hf",
        }
        return defaults.get(model_size_bucket, "meta-llama/Llama-2-7b-hf")


def build_engine_from_dataset(
    dataset_path: str | Path,
    template_dir: str | Path | None = None,
    ml_reranker_path: str | Path | None = None,
    hp_predictor_path: str | Path | None = None,
) -> ConfigRecommendationEngine:
    dataset_paths = _resolve_dataset_paths(dataset_path)
    configs, report = load_and_prepare_datasets(dataset_paths)
    profiles = build_statistical_profiles(configs)

    if template_dir is None:
        template_dir = Path(__file__).resolve().parent / "templates"
    template_engine = NotebookTemplateEngine(template_dir=template_dir)

    if len(configs) < RECOMMENDATION_MIN_SAMPLE:
        raise ValueError("insufficient normalized data to build recommendation engine")

    ml_reranker: MLReranker | None = None
    ml_reranker_status = "ml-reranker-not-configured"
    if ml_reranker_path:
        ml_reranker, ml_reranker_status = load_ml_reranker(ml_reranker_path)

    hp_predictor: HyperparameterPredictor | None = None
    hp_predictor_status = "hp-predictor-not-configured"
    if hp_predictor_path:
        hp_predictor, hp_predictor_status = load_hyperparameter_predictor(hp_predictor_path)

    return ConfigRecommendationEngine(
        configs=configs,
        profiles=profiles,
        normalization_report=report,
        template_engine=template_engine,
        ml_reranker=ml_reranker,
        ml_reranker_status=ml_reranker_status,
        hp_predictor=hp_predictor,
        hp_predictor_status=hp_predictor_status,
    )


def _resolve_dataset_paths(dataset_path: str | Path) -> list[Path]:
    raw = str(dataset_path)
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        tokens = [str(dataset_path)]

    resolved: list[Path] = []
    seen: set[Path] = set()
    for token in tokens:
        path = Path(token).expanduser()
        if any(ch in token for ch in {"*", "?", "["}):
            matches = sorted(path.parent.glob(path.name))
            for match in matches:
                absolute = match.resolve()
                if absolute in seen:
                    continue
                resolved.append(absolute)
                seen.add(absolute)
            continue

        absolute = path.resolve()
        if absolute in seen:
            continue
        resolved.append(absolute)
        seen.add(absolute)

    if len(tokens) == 1:
        primary = resolved[0]
        for extra in sorted(primary.parent.glob("*.jsonl")):
            extra_name = extra.name.lower()
            if not (
                extra_name.startswith("real_world")
                or extra_name.startswith("fixed")
            ):
                continue
            absolute = extra.resolve()
            if absolute in seen:
                continue
            resolved.append(absolute)
            seen.add(absolute)

    return resolved
