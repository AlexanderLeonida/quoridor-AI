import { useEffect, useState } from "react";
import type { ProjectStats } from "../types";

function fmtNum(n: number | undefined): string {
  if (n == null) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return n.toLocaleString();
}

interface StatCardProps {
  label: string;
  value: string;
  sub?: string;
}

function StatCard({ label, value, sub }: StatCardProps) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

export function StatsBanner() {
  const [stats, setStats] = useState<ProjectStats | null>(null);

  useEffect(() => {
    fetch("/api/project_stats")
      .then((r) => r.json())
      .then((j) => setStats(j.stats ?? {}))
      .catch(() => setStats({}));
  }, []);

  if (!stats) {
    return <div className="stats-banner stats-banner-loading" />;
  }

  const arch =
    stats.arch_blocks && stats.arch_filters
      ? `${stats.arch_blocks}× ${stats.arch_filters}f ResNet`
      : "—";

  return (
    <div className="stats-banner">
      <StatCard
        label="Self-Play Games"
        value={fmtNum(stats.games_total)}
        sub="recorded"
      />
      <StatCard
        label="Moves Played"
        value={fmtNum(stats.moves_total)}
        sub="across all games"
      />
      <StatCard
        label="Training Iters"
        value={fmtNum(stats.iterations_max)}
        sub={`${stats.promotions ?? 0} promotions`}
      />
      <StatCard
        label="Checkpoints"
        value={fmtNum(stats.checkpoints_saved)}
        sub="iterations saved"
      />
      <StatCard
        label="Net Parameters"
        value={fmtNum(stats.param_count)}
        sub={arch}
      />
    </div>
  );
}
