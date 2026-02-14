"use client";

import { useMemo, useState } from "react";

import {
  type HealthResponse,
  type RecommendPayload,
  type RecommendResponse,
  fetchRecommendation,
  submitRecommendationFeedback
} from "../lib/api";

type Props = {
  health: HealthResponse;
};

const PLATFORM_GPU_MATRIX: Record<string, string[]> = {
  colab_free: ["T4_16GB", "P100_16GB"],
  colab_pro: ["T4_16GB", "V100_16GB", "A100_40GB"],
  kaggle_free: ["T4_16GB", "P100_16GB"],
  lightning_free: ["CPU"],
  lightning_pro: ["A10G_24GB", "A100_40GB"]
};

const COMMON_TASKS = [
  "instruction_following",
  "chat",
  "text_classification",
  "question_answering",
  "summarization",
  "code_generation"
];

const PRESETS: Array<{
  id: string;
  label: string;
  description: string;
  patch: Partial<RecommendPayload>;
}> = [
  {
    id: "classify-fast",
    label: "Fast Classification",
    description: "Quick iteration baseline for text labels.",
    patch: {
      task_type: "text_classification",
      adapter_type: "lora",
      model_size_bucket: "small",
      strategy: "auto"
    }
  },
  {
    id: "qlora-chat",
    label: "QLoRA Chat",
    description: "Balanced chat tuning on limited VRAM.",
    patch: {
      task_type: "instruction_following",
      adapter_type: "qlora",
      model_size_bucket: "medium",
      strategy: "auto"
    }
  },
  {
    id: "hybrid-search",
    label: "Hybrid Explore",
    description: "Ranking-aware exploration with top-k rerank.",
    patch: {
      strategy: "hybrid",
      rerank_top_k: 5
    }
  },
  {
    id: "hybrid-ml",
    label: "Hybrid ML",
    description: "Learned reranker on top of deterministic safety filters.",
    patch: {
      strategy: "hybrid_ml",
      rerank_top_k: 5
    }
  }
];

const INITIAL_PAYLOAD: RecommendPayload = {
  platform: "colab",
  plan: "free",
  task_type: "instruction_following",
  adapter_type: "qlora",
  model_size_bucket: "medium",
  push_to_hub: false,
  strategy: "auto",
  rerank_top_k: 5,
  include_notebook: true
};

export function RecommendationForm({ health }: Props) {
  const [payload, setPayload] = useState<RecommendPayload>(INITIAL_PAYLOAD);
  const [response, setResponse] = useState<RecommendResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [flashMessage, setFlashMessage] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [feedbackLoading, setFeedbackLoading] = useState(false);

  const platformKey = `${payload.platform}_${payload.plan}`;
  const gpuOptions = useMemo(() => PLATFORM_GPU_MATRIX[platformKey] ?? [], [platformKey]);

  const responseJson = useMemo(
    () => (response ? JSON.stringify(response, null, 2) : ""),
    [response]
  );
  const installCommand = useMemo(
    () => (response ? `pip install ${response.dependency_stack.join(" ")}` : ""),
    [response]
  );

  const vramRatioPercent = useMemo(() => {
    if (!response || response.selected_gpu_memory_gb <= 0) {
      return 0;
    }
    const ratio = (response.estimated_vram_gb_per_gpu / response.selected_gpu_memory_gb) * 100;
    return Math.max(0, Math.min(100, Number(ratio.toFixed(1))));
  }, [response]);

  const vramRiskLabel = useMemo(() => {
    if (!response) {
      return "Unknown";
    }
    if (vramRatioPercent >= 90) {
      return "High";
    }
    if (vramRatioPercent >= 70) {
      return "Moderate";
    }
    return "Low";
  }, [response, vramRatioPercent]);

  function updateField<K extends keyof RecommendPayload>(key: K, value: RecommendPayload[K]) {
    setPayload((prev) => ({ ...prev, [key]: value }));
  }

  function applyPreset(patch: Partial<RecommendPayload>) {
    setPayload((prev) => ({ ...prev, ...patch }));
  }

  function setManualOverride(field: "sequence_length" | "num_gpus" | "epochs", enabled: boolean) {
    setPayload((prev) => {
      if (enabled) {
        if (field === "sequence_length") {
          return { ...prev, sequence_length: prev.sequence_length ?? 1024 };
        }
        if (field === "num_gpus") {
          return { ...prev, num_gpus: prev.num_gpus ?? 1 };
        }
        return { ...prev, epochs: prev.epochs ?? 3 };
      }

      if (field === "sequence_length") {
        const { sequence_length: _, ...rest } = prev;
        return rest;
      }
      if (field === "num_gpus") {
        const { num_gpus: _, ...rest } = prev;
        return rest;
      }
      const { epochs: _, ...rest } = prev;
      return rest;
    });
  }

  function resetForm() {
    setPayload(INITIAL_PAYLOAD);
    setShowAdvanced(false);
    setError(null);
    setResponse(null);
    setFlashMessage("Form reset");
    setTimeout(() => setFlashMessage(null), 1400);
  }

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const result = await fetchRecommendation(payload);
      setResponse(result);
    } catch (err) {
      setResponse(null);
      setError(err instanceof Error ? err.message : "Unknown recommendation error");
    } finally {
      setLoading(false);
    }
  }

  async function copyText(value: string, label: string) {
    if (!value.trim()) {
      return;
    }
    try {
      await navigator.clipboard.writeText(value);
      setFlashMessage(`${label} copied`);
      setTimeout(() => setFlashMessage(null), 1400);
    } catch {
      setFlashMessage(`Copy failed for ${label}`);
      setTimeout(() => setFlashMessage(null), 1800);
    }
  }

  async function sendFeedback(rating: number, success: boolean) {
    if (!response?.recommendation_event_id) {
      setFlashMessage("Feedback unavailable: no event id");
      setTimeout(() => setFlashMessage(null), 1800);
      return;
    }

    setFeedbackLoading(true);
    try {
      await submitRecommendationFeedback({
        recommendation_event_id: response.recommendation_event_id,
        rating,
        success
      });
      setFlashMessage("Feedback saved. Brain updated.");
      setTimeout(() => setFlashMessage(null), 1800);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Feedback failed";
      setFlashMessage(message);
      setTimeout(() => setFlashMessage(null), 2200);
    } finally {
      setFeedbackLoading(false);
    }
  }

  function downloadNotebook() {
    if (!response?.notebook_json) {
      return;
    }
    const content = JSON.stringify(response.notebook_json, null, 2);
    const blob = new Blob([content], { type: "application/x-ipynb+json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    const suffix = response.recommendation_event_id ? `_${response.recommendation_event_id}` : "";
    link.download = `recommended_config${suffix}.ipynb`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[1.04fr_1fr]">
      <form
        onSubmit={onSubmit}
        className="noise-grid glass-panel fade-up rounded-3xl border border-black/10 p-5 shadow-panel sm:p-6"
      >
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">Configuration Studio</h2>
            <p className="mt-1 text-sm text-black/65">
              Pick environment, objective, and constraints. The engine returns safe values and a notebook.
            </p>
          </div>
          <span className="mono rounded-full border border-black/15 bg-white/90 px-3 py-1 text-xs text-black/75">
            status: {health.status} · rerank: {health.ml_reranker_loaded ? "ready" : "fallback"} · hp:{" "}
            {health.hp_predictor_loaded ? "ready" : "baseline"}
          </span>
        </div>

        <section className="mb-5">
          <p className="mono mb-2 text-[11px] uppercase tracking-[0.2em] text-black/60">Quick Presets</p>
          <div className="grid gap-2 sm:grid-cols-3">
            {PRESETS.map((preset) => (
              <button
                key={preset.id}
                type="button"
                onClick={() => applyPreset(preset.patch)}
                className="rounded-xl border border-black/15 bg-white/85 p-3 text-left transition hover:-translate-y-[1px] hover:border-black/30 hover:shadow-sm"
              >
                <div className="text-sm font-semibold">{preset.label}</div>
                <div className="mt-1 text-xs text-black/65">{preset.description}</div>
              </button>
            ))}
          </div>
        </section>

        <section className="mb-5 rounded-2xl border border-black/10 bg-white/70 p-4">
          <p className="mono mb-3 text-[11px] uppercase tracking-[0.2em] text-black/60">1. Compute Environment</p>

          <div className="grid gap-3 md:grid-cols-2">
            <label className="text-sm">
              Platform
              <select
                className="mt-1 w-full rounded-xl border border-black/20 bg-white px-3 py-2"
                value={payload.platform}
                onChange={(event) => updateField("platform", event.target.value as RecommendPayload["platform"])}
              >
                <option value="colab">Google Colab</option>
                <option value="kaggle">Kaggle</option>
                <option value="lightning">Lightning AI</option>
              </select>
            </label>

            <label className="text-sm">
              Plan
              <select
                className="mt-1 w-full rounded-xl border border-black/20 bg-white px-3 py-2"
                value={payload.plan}
                onChange={(event) => updateField("plan", event.target.value as RecommendPayload["plan"])}
              >
                <option value="free">Free</option>
                <option value="pro">Pro</option>
              </select>
            </label>

            <label className="text-sm md:col-span-2">
              GPU Override
              <select
                className="mt-1 w-full rounded-xl border border-black/20 bg-white px-3 py-2"
                value={payload.gpu_override ?? ""}
                onChange={(event) => updateField("gpu_override", event.target.value || undefined)}
              >
                <option value="">Auto-select safest option</option>
                {gpuOptions.map((gpu) => (
                  <option value={gpu} key={gpu}>
                    {gpu}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </section>

        <section className="mb-5 rounded-2xl border border-black/10 bg-white/70 p-4">
          <p className="mono mb-3 text-[11px] uppercase tracking-[0.2em] text-black/60">2. Objective + Adapter</p>

          <div className="grid gap-3 md:grid-cols-2">
            <label className="text-sm md:col-span-2">
              Task
              <input
                list="task-list"
                className="mt-1 w-full rounded-xl border border-black/20 bg-white px-3 py-2"
                value={payload.task_type}
                onChange={(event) => updateField("task_type", event.target.value)}
              />
              <datalist id="task-list">
                {COMMON_TASKS.map((task) => (
                  <option key={task} value={task} />
                ))}
              </datalist>
            </label>

            <label className="text-sm">
              Adapter
              <select
                className="mt-1 w-full rounded-xl border border-black/20 bg-white px-3 py-2"
                value={payload.adapter_type}
                onChange={(event) => updateField("adapter_type", event.target.value as RecommendPayload["adapter_type"])}
              >
                <option value="none">None</option>
                <option value="lora">LoRA</option>
                <option value="qlora">QLoRA 4-bit</option>
              </select>
            </label>

            <label className="text-sm">
              Model Size
              <select
                className="mt-1 w-full rounded-xl border border-black/20 bg-white px-3 py-2"
                value={payload.model_size_bucket}
                onChange={(event) =>
                  updateField("model_size_bucket", event.target.value as RecommendPayload["model_size_bucket"])
                }
              >
                <option value="small">Small (&lt;1B)</option>
                <option value="medium">Medium (1B-7B)</option>
                <option value="large">Large (&gt;7B)</option>
              </select>
            </label>
          </div>
        </section>

        <section className="mb-4 rounded-2xl border border-black/10 bg-white/70 p-4">
          <p className="mono mb-3 text-[11px] uppercase tracking-[0.2em] text-black/60">3. Constraints + Strategy</p>

          <div className="rounded-xl border border-black/10 bg-white/85 p-3">
            <p className="text-sm font-semibold">Auto Mode (Recommended)</p>
            <p className="mt-1 text-xs text-black/65">
              Sequence Length, Num GPUs, Epochs, and Strategy are selected automatically from dataset priors,
              platform/GPU constraints, and reranker availability.
            </p>
            <button
              type="button"
              onClick={() => setShowAdvanced((prev) => !prev)}
              className="mt-3 rounded-lg border border-black/20 bg-white px-3 py-1.5 text-xs font-semibold transition hover:bg-black hover:text-white"
            >
              {showAdvanced ? "Hide Advanced Overrides" : "Show Advanced Overrides"}
            </button>
          </div>

          {showAdvanced ? (
            <div className="mt-3 grid gap-3 md:grid-cols-2">
              <label className="flex items-center gap-2 text-sm md:col-span-2">
                <input
                  type="checkbox"
                  className="h-4 w-4 accent-ember"
                  checked={payload.sequence_length !== undefined}
                  onChange={(event) => setManualOverride("sequence_length", event.target.checked)}
                />
                Manual Sequence Length
              </label>

              {payload.sequence_length !== undefined ? (
                <label className="text-sm md:col-span-2">
                  Sequence Length: <span className="mono text-xs">{payload.sequence_length}</span>
                  <input
                    type="range"
                    min={128}
                    max={4096}
                    step={64}
                    className="mt-2 w-full accent-ember"
                    value={payload.sequence_length}
                    onChange={(event) => updateField("sequence_length", Number(event.target.value))}
                  />
                </label>
              ) : null}

              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="h-4 w-4 accent-ember"
                  checked={payload.num_gpus !== undefined}
                  onChange={(event) => setManualOverride("num_gpus", event.target.checked)}
                />
                Manual Num GPUs
              </label>

              {payload.num_gpus !== undefined ? (
                <label className="text-sm">
                  Num GPUs
                  <input
                    type="number"
                    min={1}
                    max={8}
                    className="mt-1 w-full rounded-xl border border-black/20 bg-white px-3 py-2"
                    value={payload.num_gpus}
                    onChange={(event) => updateField("num_gpus", Number(event.target.value) || 1)}
                  />
                </label>
              ) : (
                <div className="text-xs text-black/60">Num GPUs will be auto-selected.</div>
              )}

              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="h-4 w-4 accent-ember"
                  checked={payload.epochs !== undefined}
                  onChange={(event) => setManualOverride("epochs", event.target.checked)}
                />
                Manual Epochs
              </label>

              {payload.epochs !== undefined ? (
                <label className="text-sm">
                  Epochs
                  <input
                    type="number"
                    step="0.1"
                    min={0.1}
                    max={20}
                    className="mt-1 w-full rounded-xl border border-black/20 bg-white px-3 py-2"
                    value={payload.epochs}
                    onChange={(event) => updateField("epochs", Number(event.target.value) || 3)}
                  />
                </label>
              ) : (
                <div className="text-xs text-black/60">Epochs will be auto-selected.</div>
              )}

              <label className="text-sm">
                Strategy
                <select
                  className="mt-1 w-full rounded-xl border border-black/20 bg-white px-3 py-2"
                  value={payload.strategy}
                  onChange={(event) => updateField("strategy", event.target.value as RecommendPayload["strategy"])}
                >
                  <option value="auto">Auto (Recommended)</option>
                  <option value="deterministic">Deterministic Baseline</option>
                  <option value="hybrid">Hybrid Rerank</option>
                  <option value="hybrid_ml">Hybrid ML Rerank</option>
                </select>
              </label>

              <label className="text-sm">
                Rerank Top-K
                <input
                  type="number"
                  min={1}
                  max={15}
                  className="mt-1 w-full rounded-xl border border-black/20 bg-white px-3 py-2"
                  value={payload.rerank_top_k}
                  onChange={(event) => updateField("rerank_top_k", Number(event.target.value) || 5)}
                />
              </label>
            </div>
          ) : null}

          <div className="mt-3 grid gap-3 md:grid-cols-2">
            <label className="text-sm md:col-span-2">
              Dataset Name (optional)
              <input
                className="mt-1 w-full rounded-xl border border-black/20 bg-white px-3 py-2"
                value={payload.dataset_name ?? ""}
                onChange={(event) => updateField("dataset_name", event.target.value || undefined)}
                placeholder="alpaca, squad, custom_dataset..."
              />
            </label>

            <label className="flex items-center gap-2 text-sm md:col-span-2">
              <input
                type="checkbox"
                className="h-4 w-4 accent-ember"
                checked={payload.include_notebook}
                onChange={(event) => updateField("include_notebook", event.target.checked)}
              />
              Include notebook payload for direct `.ipynb` download
            </label>
          </div>
        </section>

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <button
            type="submit"
            disabled={loading}
            className="rounded-xl bg-ember px-4 py-2 text-sm font-semibold text-white transition hover:brightness-95 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? "Generating Recommendation..." : "Generate Recommendation"}
          </button>
          <button
            type="button"
            onClick={resetForm}
            className="rounded-xl border border-black/20 bg-white/90 px-4 py-2 text-sm font-semibold text-black/80 transition hover:bg-black hover:text-white"
          >
            Reset
          </button>
          <span className="mono text-xs text-black/60">db: {health.database_url}</span>
          {flashMessage ? <span className="mono text-xs text-teal">{flashMessage}</span> : null}
        </div>

        {error ? <p className="mt-4 rounded-xl bg-red-100 p-3 text-sm text-red-700">{error}</p> : null}
      </form>

      <section className="glass-panel fade-up rounded-3xl border border-black/10 p-5 shadow-panel sm:p-6">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-2xl font-semibold tracking-tight">Recommendation Outcome</h2>
          <span className="mono rounded-full border border-black/15 bg-white/85 px-3 py-1 text-xs text-black/70">
            {response ? "generated" : "waiting"}
          </span>
        </div>

        {!response ? (
          <div className="space-y-4 rounded-2xl border border-black/10 bg-white/70 p-4">
            <p className="text-sm text-black/70">
              Submit the form to generate a GPU-safe configuration and notebook package.
            </p>
            <div className="grid gap-2 sm:grid-cols-3">
              <HintCard title="Safe Hyperparams" subtitle="Clamped to hardware limits" />
              <HintCard title="Dependency Stack" subtitle="Pinned versions by platform" />
              <HintCard title="Notebook Output" subtitle="Template-driven reproducible runbook" />
            </div>
          </div>
        ) : (
          <div className="space-y-5">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <MetricCard label="Selected GPU" value={response.selected_gpu} />
              <MetricCard label="Strategy" value={response.recommendation_basis.strategy} />
              <MetricCard label="VRAM Est." value={`${response.estimated_vram_gb_per_gpu} GB`} />
              <MetricCard label="Train Time" value={`${response.estimated_training_time_hours} h`} />
              <MetricCard
                label="Confidence"
                value={`${Math.round((response.recommendation_basis.confidence_score ?? 0) * 100)}% (${response.recommendation_basis.confidence_level ?? "n/a"})`}
              />
            </div>

            <div className="rounded-2xl border border-black/10 bg-white/70 p-4">
              <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-black/70">Resolved Constraints</h3>
              <div className="grid gap-2 sm:grid-cols-3">
                <div className="rounded-lg border border-black/10 bg-white/90 px-3 py-2">
                  <div className="mono text-[11px] uppercase tracking-wide text-black/55">Sequence Length</div>
                  <div className="mono mt-1 text-sm text-black/85">
                    {String(response.recommendation_basis.resolved_sequence_length ?? response.safe_hyperparameters.max_seq_length)}
                  </div>
                </div>
                <div className="rounded-lg border border-black/10 bg-white/90 px-3 py-2">
                  <div className="mono text-[11px] uppercase tracking-wide text-black/55">Num GPUs</div>
                  <div className="mono mt-1 text-sm text-black/85">
                    {String(response.recommendation_basis.resolved_num_gpus ?? response.safe_hyperparameters.num_gpus)}
                  </div>
                </div>
                <div className="rounded-lg border border-black/10 bg-white/90 px-3 py-2">
                  <div className="mono text-[11px] uppercase tracking-wide text-black/55">Epochs</div>
                  <div className="mono mt-1 text-sm text-black/85">
                    {String(response.recommendation_basis.resolved_epochs ?? response.safe_hyperparameters.epochs)}
                  </div>
                </div>
              </div>
            </div>

            {response.recommendation_basis.hp_predictor ? (
              <div className="rounded-2xl border border-black/10 bg-white/70 p-4">
                <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-black/70">
                  HP Predictor
                </h3>
                <p className="mono text-xs text-black/65">
                  {response.recommendation_basis.hp_predictor.loaded ? "loaded" : "unavailable"} ·{" "}
                  {response.recommendation_basis.hp_predictor.status}
                </p>
              </div>
            ) : null}

            {response.recommendation_basis.requested_task_type &&
            response.recommendation_basis.resolved_task_type &&
            response.recommendation_basis.requested_task_type !== response.recommendation_basis.resolved_task_type ? (
              <div className="rounded-2xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                Task normalized from <span className="mono">{response.recommendation_basis.requested_task_type}</span>{" "}
                to <span className="mono">{response.recommendation_basis.resolved_task_type}</span> for profile matching.
              </div>
            ) : null}

            <div className="rounded-2xl border border-black/10 bg-white/70 p-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold uppercase tracking-wide text-black/70">Safety Meter</h3>
                <span className="mono text-xs text-black/70">
                  VRAM utilization {vramRatioPercent}% · risk {vramRiskLabel}
                </span>
              </div>
              <div className="meter-track mt-2">
                <div className="meter-fill" style={{ width: `${vramRatioPercent}%` }} />
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => copyText(responseJson, "Recommendation JSON")}
                className="rounded-lg border border-black/20 bg-white px-3 py-2 text-xs font-semibold transition hover:bg-black hover:text-white"
              >
                Copy Recommendation JSON
              </button>
              <button
                type="button"
                onClick={() => copyText(installCommand, "Dependency install command")}
                className="rounded-lg border border-black/20 bg-white px-3 py-2 text-xs font-semibold transition hover:bg-black hover:text-white"
              >
                Copy Install Command
              </button>
              <button
                type="button"
                onClick={downloadNotebook}
                disabled={!response.notebook_json}
                className="rounded-lg border border-black/20 bg-white px-3 py-2 text-xs font-semibold transition hover:bg-black hover:text-white disabled:cursor-not-allowed disabled:opacity-45"
              >
                Download .ipynb
              </button>
              <button
                type="button"
                onClick={() => sendFeedback(5, true)}
                disabled={feedbackLoading || !response.recommendation_event_id}
                className="rounded-lg border border-emerald-300 bg-emerald-50 px-3 py-2 text-xs font-semibold text-emerald-800 transition hover:bg-emerald-600 hover:text-white disabled:cursor-not-allowed disabled:opacity-45"
              >
                Good Suggestion
              </button>
              <button
                type="button"
                onClick={() => sendFeedback(2, false)}
                disabled={feedbackLoading || !response.recommendation_event_id}
                className="rounded-lg border border-rose-300 bg-rose-50 px-3 py-2 text-xs font-semibold text-rose-800 transition hover:bg-rose-600 hover:text-white disabled:cursor-not-allowed disabled:opacity-45"
              >
                Needs Improvement
              </button>
            </div>

            <div className="rounded-2xl border border-black/10 bg-white/70 p-4">
              <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-black/70">Safe Hyperparameters</h3>
              <div className="grid gap-2 sm:grid-cols-2">
                {Object.entries(response.safe_hyperparameters).map(([key, value]) => (
                  <div key={key} className="rounded-lg border border-black/10 bg-white/90 px-3 py-2">
                    <div className="mono text-[11px] uppercase tracking-wide text-black/55">{key}</div>
                    <div className="mono mt-1 text-sm text-black/85">{String(value)}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-2xl border border-black/10 bg-white/70 p-4">
              <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-black/70">Dependency Stack</h3>
              <div className="grid gap-1">
                {response.dependency_stack.map((pkg) => (
                  <div key={pkg} className="mono text-xs text-black/80">
                    {pkg}
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-2xl border border-black/10 bg-white/70 p-4">
              <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-black/70">Top Ranked Candidates</h3>
              {response.recommendation_basis.ranked_candidates.length === 0 ? (
                <p className="text-sm text-black/65">
                  Deterministic mode does not produce ranked candidates. Switch to hybrid for ranking diagnostics.
                </p>
              ) : (
                <div className="space-y-2">
                  {response.recommendation_basis.ranked_candidates.map((row, index) => (
                    <div
                      key={`${row.record_id ?? index}`}
                      className="rounded-xl border border-black/10 bg-white/90 px-3 py-2 text-xs"
                    >
                      <div className="flex items-center justify-between">
                        <span className="mono font-semibold">#{index + 1}</span>
                        <span className="mono text-black/65">score: {String(row.total_score)}</span>
                      </div>
                      <div className="mono mt-1 break-all text-black/75">
                        {String(row.model_name)} | {String(row.dataset_name)}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {response.notes.length > 0 ? (
              <div className="rounded-2xl border border-black/10 bg-white/70 p-4">
                <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-black/70">Notes</h3>
                <ul className="list-disc space-y-1 pl-5 text-sm">
                  {response.notes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        )}
      </section>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-black/10 bg-white/85 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-black/55">{label}</div>
      <div className="mono mt-1 text-sm font-semibold text-black/90">{value}</div>
    </div>
  );
}

function HintCard({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="rounded-xl border border-black/10 bg-white/85 px-3 py-2">
      <div className="text-sm font-semibold text-black/85">{title}</div>
      <div className="mt-1 text-xs text-black/60">{subtitle}</div>
    </div>
  );
}
