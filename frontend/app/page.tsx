import { RecommendationForm } from "../components/recommendation-form";
import {
  fetchHealth,
  fetchRecentRecommendations,
  type HealthResponse,
  type RecommendationEvent
} from "../lib/api";

const FALLBACK_HEALTH: HealthResponse = {
  status: "unreachable",
  dataset_path: "unknown",
  database_url: "unknown",
  normalized_configs: 0,
  profiles: 0,
  sync_stats: {}
};

const FALLBACK_RECENT: RecommendationEvent[] = [];

export default async function HomePage() {
  let health = FALLBACK_HEALTH;
  let recent = FALLBACK_RECENT;
  let healthError: string | null = null;
  let recentError: string | null = null;

  try {
    health = await fetchHealth();
  } catch (error) {
    healthError = error instanceof Error ? error.message : "Failed to connect to backend";
  }

  try {
    recent = await fetchRecentRecommendations(6);
  } catch (error) {
    recentError = error instanceof Error ? error.message : "Failed to load recent recommendations";
  }

  return (
    <main className="mx-auto flex w-full max-w-7xl flex-col gap-8 px-4 py-8 sm:px-6">
      <header className="glass-panel relative overflow-hidden rounded-3xl border border-black/10 p-6 shadow-panel sm:p-8">
        <div className="pulse-orb absolute -right-12 -top-16 h-52 w-52 rounded-full bg-gradient-to-br from-ember/35 to-marine/15 blur-2xl" />
        <div className="relative grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
          <div>
            <p className="mono text-xs uppercase tracking-[0.26em] text-black/55">Fine-Tune Configuration OS</p>
            <h1 className="mt-2 text-3xl font-bold leading-tight sm:text-5xl">
              Build Safer Fine-Tuning Runs In Minutes
            </h1>
            <p className="mt-3 max-w-2xl text-sm text-black/70 sm:text-base">
              Platform-aware recommendation intelligence with deterministic baselines, GPU safety guards, and
              reproducible notebook generation.
            </p>

            <div className="mt-5 grid gap-3 sm:grid-cols-3">
              <StatTile label="Configs" value={String(health.normalized_configs)} />
              <StatTile label="Profiles" value={String(health.profiles)} />
              <StatTile
                label="Auth"
                value={health.auth_required ? "Required" : "Open"}
                hint={`Rate ${health.rate_limit_per_minute ?? "-"} / min`}
              />
            </div>

            {healthError ? (
              <p className="mt-4 rounded-xl border border-amber-300 bg-amber-100/85 px-3 py-2 text-sm text-amber-900">
                Backend health fetch failed: {healthError}
              </p>
            ) : null}
          </div>

          <aside className="rounded-2xl border border-black/15 bg-white/75 p-4 shadow-inner">
            <div className="flex items-center justify-between">
              <p className="mono text-xs uppercase tracking-[0.2em] text-black/60">Recent Activity</p>
              <span className="mono text-xs text-black/55">{recent.length} runs</span>
            </div>

            {recent.length === 0 ? (
              <p className="mt-3 text-sm text-black/65">
                {recentError ?? "No recent recommendations logged yet."}
              </p>
            ) : (
              <div className="mt-3 space-y-2">
                {recent.map((item) => (
                  <div
                    key={item.id}
                    className="fade-up rounded-xl border border-black/10 bg-white/85 px-3 py-2 text-xs"
                  >
                    <div className="flex items-center justify-between">
                      <span className="mono text-black/70">#{item.id}</span>
                      <span className="mono text-black/55">{new Date(item.created_at).toLocaleString()}</span>
                    </div>
                    <div className="mt-1 font-semibold text-black/85">
                      {item.selected_gpu} · {item.strategy}
                    </div>
                    <div className="mono mt-1 text-black/65">
                      {item.estimated_vram_gb_per_gpu} GB · {item.estimated_training_time_hours} h
                    </div>
                  </div>
                ))}
              </div>
            )}
          </aside>
        </div>
      </header>

      <section>
        <RecommendationForm health={health} />
      </section>
    </main>
  );
}

function StatTile({
  label,
  value,
  hint
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-xl border border-black/15 bg-white/80 px-3 py-2">
      <div className="mono text-[11px] uppercase tracking-wide text-black/60">{label}</div>
      <div className="mt-1 text-xl font-semibold leading-none text-black/90">{value}</div>
      {hint ? <div className="mono mt-1 text-[11px] text-black/55">{hint}</div> : null}
    </div>
  );
}
