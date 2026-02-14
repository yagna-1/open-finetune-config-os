export type RecommendPayload = {
  platform: "colab" | "kaggle" | "lightning";
  plan: "free" | "pro";
  task_type: string;
  adapter_type: "none" | "lora" | "qlora";
  model_size_bucket?: "small" | "medium" | "large";
  model_name?: string;
  model_parameter_count?: string;
  dataset_name?: string;
  dataset_size?: number;
  sequence_length?: number;
  num_gpus?: number;
  gpu_override?: string;
  epochs?: number;
  push_to_hub: boolean;
  huggingface_repo_id?: string;
  strategy: "auto" | "hybrid" | "deterministic" | "hybrid_ml";
  rerank_top_k: number;
  include_notebook: boolean;
};

export type RecommendResponse = {
  platform_key: string;
  selected_gpu: string;
  selected_gpu_memory_gb: number;
  safe_hyperparameters: Record<string, string | number | null>;
  dependency_stack: string[];
  estimated_vram_gb_per_gpu: number;
  estimated_training_time_hours: number;
  recommendation_basis: {
    strategy: string;
    requested_strategy?: string;
    rerank_top_k: number;
    requested_sequence_length?: number | null;
    requested_num_gpus?: number | null;
    requested_epochs?: number | null;
    resolved_sequence_length?: number;
    resolved_num_gpus?: number;
    resolved_epochs?: number;
    requested_model_size_bucket?: string;
    requested_adapter_type?: string;
    resolved_adapter_type?: string;
    candidate_pool_match_type?: string;
    brain_strategy_hint?: string | null;
    brain_decision?: Record<string, unknown>;
    requested_task_type?: string;
    resolved_task_type?: string;
    resolved_parameter_count?: number;
    profile_match_type: string;
    profile_sample_size: number;
    candidate_pool_size: number;
    gpu_safe_pool_size: number;
    confidence_score?: number;
    confidence_level?: string;
    ml_reranker?: {
      loaded: boolean;
      status: string;
      model_version?: string | null;
    };
    hp_predictor?: {
      loaded: boolean;
      status: string;
      model_version?: string | null;
      prediction?: Record<string, Record<string, number>>;
    };
    ranked_candidates: Array<Record<string, string | number | null>>;
  };
  notebook_template: string;
  notebook_json?: Record<string, unknown> | null;
  notes: string[];
  recommendation_event_id?: number;
};

export type RecommendationEvent = {
  id: number;
  strategy: string;
  selected_gpu: string;
  estimated_vram_gb_per_gpu: number;
  estimated_training_time_hours: number;
  created_at: string;
  request_payload: Record<string, unknown>;
  result_payload: RecommendResponse;
};

export type HealthResponse = {
  status: string;
  dataset_path: string;
  database_url: string;
  normalized_configs: number;
  profiles: number;
  sync_stats: Record<string, number>;
  auth_required?: boolean;
  rate_limit_per_minute?: number;
  ml_reranker_loaded?: boolean;
  ml_reranker_status?: string;
  ml_reranker_model_version?: string | null;
  hp_predictor_loaded?: boolean;
  hp_predictor_status?: string;
  hp_predictor_model_version?: string | null;
};

export type FeedbackPayload = {
  recommendation_event_id: number;
  rating?: number;
  success?: boolean;
  notes?: string;
};

export type FeedbackResponse = {
  status: string;
  feedback: {
    id: number;
    recommendation_event_id: number;
    rating?: number | null;
    success?: boolean | null;
    notes?: string | null;
    created_at: string;
  };
};

function resolveApiBaseUrl(): string {
  if (typeof window === "undefined") {
    return process.env.API_BASE_URL_SERVER ?? process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
  }
  return process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
}

function resolveApiKey(): string | undefined {
  if (typeof window === "undefined") {
    return process.env.API_KEY_SERVER ?? process.env.NEXT_PUBLIC_API_KEY;
  }
  return process.env.NEXT_PUBLIC_API_KEY;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const apiKey = resolveApiKey();
  const headers: Record<string, string> = {};
  if (apiKey) {
    headers["x-api-key"] = apiKey;
  }
  const response = await fetch(`${resolveApiBaseUrl()}/health`, {
    cache: "no-store",
    headers
  });
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }
  return (await response.json()) as HealthResponse;
}

export async function fetchRecentRecommendations(limit = 8): Promise<RecommendationEvent[]> {
  const apiKey = resolveApiKey();
  const headers: Record<string, string> = {};
  if (apiKey) {
    headers["x-api-key"] = apiKey;
  }

  const response = await fetch(`${resolveApiBaseUrl()}/recommendations/recent?limit=${limit}`, {
    cache: "no-store",
    headers
  });

  if (!response.ok) {
    throw new Error(`Recent recommendations fetch failed: ${response.status}`);
  }

  const payload = (await response.json()) as { items?: RecommendationEvent[] };
  return payload.items ?? [];
}

export async function fetchRecommendation(payload: RecommendPayload): Promise<RecommendResponse> {
  const apiKey = resolveApiKey();
  const headers: Record<string, string> = {
    "content-type": "application/json"
  };
  if (apiKey) {
    headers["x-api-key"] = apiKey;
  }
  const response = await fetch(`${resolveApiBaseUrl()}/recommend`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`Recommendation failed (${response.status}): ${errorBody}`);
  }

  return (await response.json()) as RecommendResponse;
}

export async function submitRecommendationFeedback(payload: FeedbackPayload): Promise<FeedbackResponse> {
  const apiKey = resolveApiKey();
  const headers: Record<string, string> = {
    "content-type": "application/json"
  };
  if (apiKey) {
    headers["x-api-key"] = apiKey;
  }

  const response = await fetch(`${resolveApiBaseUrl()}/feedback`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`Feedback submit failed (${response.status}): ${errorBody}`);
  }
  return (await response.json()) as FeedbackResponse;
}
