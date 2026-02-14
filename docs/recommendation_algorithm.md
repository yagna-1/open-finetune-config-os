# Recommendation Algorithm (Deterministic + Hybrid + Hybrid ML)

## Pipeline

1. Load `finetuning_configs_final.jsonl`.
   - Optionally merge additional local files (e.g. `real_world*.jsonl`, `fixed*.jsonl`) when present.
   - Corrupted all-null files are ignored safely.
2. Normalize fields:
   - Accept canonical schema and real-world schema variants (e.g. top-level `task_type`, `model.size_params`, `training_config.batch_size`, `hardware_config`).
   - `effective_batch_size = batch_size_per_device * gradient_accumulation_steps * num_gpus`
   - canonicalize task aliases (e.g. `language finetuning` -> `instruction_following`)
   - `model_size_bucket`: `small (<1B)`, `medium (1B-7B)`, `large (>7B)`
   - `adapter_type`: `none | lora | qlora`
   - dataset size bucket via known-map + size thresholds
   - optimizer/scheduler/precision canonicalization
3. Reject invalid rows.
4. Deduplicate by:
   - `(model_name, dataset_name, task_type, adapter_type, learning_rate, effective_batch_size)`
5. Build grouped statistical profiles by:
   - `(task_type, model_size_bucket, adapter_type)`
   - median/IQR learning rate
   - median effective batch size
   - median LoRA rank
   - typical optimizer/precision

## Online Recommendation

Input:
- `platform`, `plan`, optional `gpu_override`
- `task_type`, `adapter_type`, model information
- `strategy`: `auto` (default), `deterministic`, `hybrid`, or `hybrid_ml`

Steps:
1. Resolve GPU from platform-plan matrix.
2. Filter normalized configs by `(task_type, model_size_bucket, adapter_type)`.
3. Apply profile fallback hierarchy if exact group is missing.
4. Remove GPU-incompatible configs via VRAM estimator.
5. Build baseline:
   - `auto`: prefer `hybrid_ml` when ML reranker is loaded, otherwise `hybrid`, then `deterministic` if ranking pool is unavailable.
   - `deterministic`: medians/modes from safe pool.
   - `hybrid`: rank candidates using weighted similarity + performance + efficiency, then aggregate top-k.
   - `hybrid_ml`: run heuristic ranking first, then apply trained ML reranker (if model artifact is loaded) and aggregate top-k.
     - If no model artifact is available, fallback to `hybrid`.
   - Direct HP predictor (if artifact loaded): predicts LR/batch/seq/LoRA rank with uncertainty; predictions are blended conservatively with deterministic baseline.
6. Clamp for safety:
   - sequence length by GPU tier
   - LoRA rank by GPU tier
   - batch shape by iterative OOM-safe search
   - apply conservative caps when profile fallback confidence is low
   - apply strict OOD caps when task has no direct corpus support (`batch_size_per_device <= 4`, `max_seq_length <= 512`)
7. Return:
   - safe hyperparameters
   - dependency stack (platform pinned)
   - VRAM estimate per GPU
   - training-time estimate
   - ranked-candidate diagnostics (hybrid / hybrid_ml mode)
   - ML reranker status (loaded/missing/version)
   - learning-brain decision diagnostics (when `strategy=auto`)
   - recommendation confidence score/level
   - unsupported-task and OOD-guard diagnostics
   - rendered notebook from fixed Jinja template

## Learning Brain

- Every recommendation is logged with context + output payload.
- Users can submit feedback for a recommendation event (`rating`, `success`, optional `notes`).
- Run telemetry can be ingested with start/complete events, including outcome, wall time, peak VRAM, throughput, and cost.
- For `strategy=auto`, the system consults historical events + feedback for the same `(task_type, adapter_type, model_size_bucket)` context.
- If feedback volume is sufficient, it selects the best strategy (`deterministic` / `hybrid` / `hybrid_ml`) using a weighted score over:
  - user rating and success feedback
  - historical confidence
  - safety/risk profile (VRAM utilization)
- If feedback is insufficient, it falls back to non-learning auto behavior.

## ML Reranker Training

- Script: `scripts/train_ml_reranker.py`
- Inputs: normalized config corpus (same dataset sources used by runtime engine)
- Model: sparse linear regressor over pairwise ranking features
- Outputs:
  - model artifact: `artifacts/ml_reranker.joblib`
  - metrics: `artifacts/ml_reranker_metrics.json`
- Primary offline metrics:
  - `ndcg_at_5`
  - `top1_regret`
  - `oom_violation_rate`

## Hyperparameter Predictor Training

- Script: `scripts/train_hp_predictor.py`
- Inputs: normalized config corpus (same dataset sources used by runtime engine)
- Model: per-target random-forest regressors with uncertainty estimated from validation residuals
- Predicted targets:
  - `learning_rate`
  - `effective_batch_size`
  - `max_seq_length`
  - `lora_rank` (when adapter labels exist)
- Runtime behavior:
  - predicts with confidence intervals
  - blends prediction with deterministic baseline
  - always followed by hard safety clamps (OOM prevention takes precedence)

## Memory Model

- fp16/bf16 base: `params * 2 bytes`
- qlora base: `params * 0.5 bytes`
- activations: `~30%` scaled by batch and sequence
- optimizer states: multiplier by adapter type
- safety buffer: `+20%`

## Production Controls

- Evaluation and release gates:
  - `src/ft_config_engine/evaluation/harness.py`
  - `src/ft_config_engine/evaluation/gates.py`
  - `evaluation/golden_dataset.jsonl`
- Governance and rollback:
  - `src/ft_config_engine/governance.py`
  - `scripts/promote_model.py`
  - `scripts/rollback_model.py`
- Retraining trigger loop:
  - `src/ft_config_engine/retraining.py`
  - `scripts/check_retraining.py`
- Observability/canary API controls:
  - `/metrics`
  - `/governance/*`
  - `/retraining/*`
  - `X-Model-Slot` and `X-Model-Version` response headers
