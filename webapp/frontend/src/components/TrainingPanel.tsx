import {
  Area,
  AreaChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useWebSocketStream } from "../useWebSocket";
import type { MetricsState } from "../types";

const AXIS = "#aaaaaa";
const GRID = "#3a3a4d";

function num(v: unknown): number | null {
  if (typeof v === "number" && !Number.isNaN(v)) return v;
  return null;
}

interface Props {
  /** "strip" = 1x4 horizontal row (compact, for dashboard).
   *  "grid"  = 2x2 grid (large, for solo view). */
  layout?: "strip" | "grid";
}

export function TrainingPanel({ layout = "grid" }: Props) {
  const { data, status } = useWebSocketStream<MetricsState>("metrics");

  const rows = data?.rows ?? [];
  const chartRows = rows.map((r, i) => ({
    iter: num(r.global_iter) ?? i + 1,
    train: num(r.train_loss),
    policy: num(r.policy_loss),
    value: num(r.value_loss),
    p1: num(r.sp_p1_wins) ?? 0,
    p2: num(r.sp_p2_wins) ?? 0,
    draws: num(r.sp_draws) ?? 0,
    plies: num(r.sp_avg_plies),
    evalScore: num(r.eval_score),
    promoted: num(r.promoted) === 1,
  }));

  const promotionPoints = chartRows
    .filter((r) => r.promoted && r.evalScore != null)
    .map((r) => ({ iter: r.iter, evalScore: r.evalScore }));

  // Reconstruct a running Elo from the gating record.  When a candidate is
  // promoted with win-rate `p` over the prior champion, classical Elo theory
  // says the candidate is stronger by  −400 * log10(1/p − 1)  rating points.
  // Non-promotion iterations leave Elo unchanged (champion didn't move).
  let elo = 1000;
  const eloHistory = chartRows.map((r) => {
    if (
      r.promoted &&
      r.evalScore != null &&
      r.evalScore > 0.01 &&
      r.evalScore < 0.99
    ) {
      elo += -400 * Math.log10(1 / r.evalScore - 1);
    }
    return { iter: r.iter, elo: Math.round(elo) };
  });
  const eloGain = Math.round(elo - 1000);

  const noData = chartRows.length === 0;
  const tooltipStyle = { background: "#1a1a24", border: "1px solid #444" };

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <h2>Training Dashboard</h2>
          <div className="panel-subtitle">
            Live metrics from logs/metrics.csv · ● = champion replaced
          </div>
        </div>
        <span className={`pill pill-${status}`}>
          {noData ? "no metrics" : `${chartRows.length} iterations`}
        </span>
      </div>

      {noData ? (
        <div className="panel-empty">
          waiting for <code>logs/metrics.csv</code>…
        </div>
      ) : (
        <div className={`charts-grid charts-grid-${layout}`}>
          <div className="chart-cell">
            <div className="chart-title">Training Loss</div>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={chartRows}
                margin={{ top: 4, right: 12, left: 0, bottom: 4 }}
              >
                <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
                <XAxis dataKey="iter" stroke={AXIS} fontSize={10} />
                <YAxis stroke={AXIS} fontSize={10} />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Line
                  type="monotone"
                  dataKey="train"
                  stroke="#f1c40f"
                  dot={false}
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="policy"
                  stroke="#3498db"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#e74c3c"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-cell">
            <div className="chart-title">Self-Play Outcomes / Iter</div>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={chartRows}
                margin={{ top: 4, right: 12, left: 0, bottom: 4 }}
              >
                <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
                <XAxis dataKey="iter" stroke={AXIS} fontSize={10} />
                <YAxis stroke={AXIS} fontSize={10} />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Area
                  type="monotone"
                  dataKey="p1"
                  stackId="1"
                  stroke="#e74c3c"
                  fill="#e74c3c"
                  fillOpacity={0.75}
                />
                <Area
                  type="monotone"
                  dataKey="draws"
                  stackId="1"
                  stroke="#888"
                  fill="#888"
                  fillOpacity={0.7}
                />
                <Area
                  type="monotone"
                  dataKey="p2"
                  stackId="1"
                  stroke="#3498db"
                  fill="#3498db"
                  fillOpacity={0.75}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-cell">
            <div className="chart-title">Avg Game Length (plies)</div>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={chartRows}
                margin={{ top: 4, right: 12, left: 0, bottom: 4 }}
              >
                <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
                <XAxis dataKey="iter" stroke={AXIS} fontSize={10} />
                <YAxis stroke={AXIS} fontSize={10} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line
                  type="monotone"
                  dataKey="plies"
                  stroke="#1abc9c"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-cell">
            <div className="chart-title">
              Gating Eval Score (● promoted)
            </div>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={chartRows}
                margin={{ top: 4, right: 12, left: 0, bottom: 4 }}
              >
                <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
                <XAxis dataKey="iter" stroke={AXIS} fontSize={10} />
                <YAxis
                  domain={[0, 1]}
                  stroke={AXIS}
                  fontSize={10}
                  ticks={[0, 0.25, 0.5, 0.75, 1]}
                />
                <Tooltip contentStyle={tooltipStyle} />
                <ReferenceLine
                  y={0.5}
                  stroke="#666"
                  strokeDasharray="4 4"
                />
                <Line
                  type="monotone"
                  dataKey="evalScore"
                  stroke="#f39c12"
                  strokeWidth={2}
                  dot={false}
                />
                <Scatter
                  data={promotionPoints}
                  dataKey="evalScore"
                  fill="#2ecc71"
                  shape="circle"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-cell">
            <div className="chart-title">
              Champion Elo over Training{" "}
              <span className="chart-title-sub">
                ({eloGain >= 0 ? "+" : ""}
                {eloGain} from start)
              </span>
            </div>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={eloHistory}
                margin={{ top: 4, right: 12, left: 0, bottom: 4 }}
              >
                <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
                <XAxis dataKey="iter" stroke={AXIS} fontSize={10} />
                <YAxis
                  stroke={AXIS}
                  fontSize={10}
                  domain={["dataMin - 20", "dataMax + 20"]}
                />
                <Tooltip contentStyle={tooltipStyle} />
                <Line
                  type="stepAfter"
                  dataKey="elo"
                  stroke="#9b59b6"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
