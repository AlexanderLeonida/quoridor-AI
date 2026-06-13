import { useWebSocketStream } from "../useWebSocket";
import type { SpectatorState } from "../types";
import { Board } from "./Board";

interface Props {
  streamName: string;
  fallbackTitle: string;
  /** Dense = used in the multi-column dashboard. Trims chrome to save space. */
  dense?: boolean;
}

export function SpectatorPanel({ streamName, fallbackTitle, dense }: Props) {
  const { data, status } = useWebSocketStream<SpectatorState>(streamName);

  const subtitle =
    streamName === "selfplay"
      ? "Champion playing itself · this is how training data is created"
      : streamName === "generations"
        ? "Untrained network vs trained champion · the gap RL closed"
        : undefined;

  if (!data) {
    return (
      <div className={`panel ${dense ? "panel-dense" : ""}`}>
        <div className="panel-header">
          <div className="panel-title">
            <h2>{fallbackTitle}</h2>
            {subtitle && <div className="panel-subtitle">{subtitle}</div>}
          </div>
          <span className={`pill pill-${status}`}>{status}</span>
        </div>
        <div className="panel-empty">waiting for data…</div>
      </div>
    );
  }

  const p1Adv = data.board.turn === 1 ? -data.value : data.value;
  const gaugePct = Math.max(-1, Math.min(1, p1Adv));
  const gaugeWidth = `${Math.abs(gaugePct) * 50}%`;
  const gaugeColor = gaugePct >= 0 ? "var(--p1)" : "var(--p2)";
  // Convert net's value estimate to a win probability for each side.
  // Champion-level MCTS values are already calibrated in [-1, 1] as
  // expected outcome under the current policy, so (v+1)/2 ≈ P(P1 wins).
  const p1Prob = Math.round(((p1Adv + 1) / 2) * 100);
  const p2Prob = 100 - p1Prob;

  return (
    <div className={`panel spectator-panel ${dense ? "panel-dense" : ""}`}>
      <div className="panel-header">
        <div className="panel-title">
          <h2>{data.title}</h2>
          {subtitle && <div className="panel-subtitle">{subtitle}</div>}
        </div>
        <span className={`pill pill-${status}`}>game #{data.game_num}</span>
      </div>

      <div className="nameplate top">
        <span className="dot" style={{ background: "var(--p2)" }} />
        <span className="nameplate-name">{data.p2_name}</span>
        <span className="nameplate-tag">walls {data.board.walls_left[1]}</span>
      </div>

      <div className="board-wrap">
        <Board state={data.board} />
      </div>

      <div className="nameplate bottom">
        <span className="dot" style={{ background: "var(--p1)" }} />
        <span className="nameplate-name">{data.p1_name}</span>
        <span className="nameplate-tag">walls {data.board.walls_left[0]}</span>
      </div>

      <div className="score-row">
        <span className="score-side score-p1">
          {data.p1_name}: <b>{data.score[0]}</b>
        </span>
        <span className="score-draws">draws {data.score[2]}</span>
        <span className="score-side score-p2">
          <b>{data.score[1]}</b> :{data.p2_name}
        </span>
      </div>

      <div className="win-prob">
        <div className="win-prob-row">
          <span className="win-prob-name win-prob-p1">{data.p1_name}</span>
          <span className="win-prob-pct win-prob-pct-p1">{p1Prob}%</span>
          <span className="win-prob-divider">·</span>
          <span className="win-prob-pct win-prob-pct-p2">{p2Prob}%</span>
          <span className="win-prob-name win-prob-p2">{data.p2_name}</span>
        </div>
        <div className="gauge-track">
          <div className="gauge-midline" />
          <div
            className="gauge-fill"
            style={{
              left: gaugePct >= 0 ? "50%" : `${50 + gaugePct * 50}%`,
              width: gaugeWidth,
              background: gaugeColor,
            }}
          />
        </div>
      </div>

      {data.top_moves && data.top_moves.length > 0 && (
        <div className="top-moves">
          <div className="top-moves-label">
            MCTS top moves <span className="top-moves-hint">(visit %)</span>
          </div>
          <div className="top-moves-list">
            {data.top_moves.slice(0, dense ? 3 : 5).map((m, i) => (
              <div className="top-move-row" key={`${m.notation}-${i}`}>
                <span className="top-move-rank">{i + 1}</span>
                <span className="top-move-notation">{m.notation}</span>
                <div className="top-move-bar">
                  <div
                    className="top-move-bar-fill"
                    style={{ width: `${Math.round(m.weight * 100)}%` }}
                  />
                </div>
                <span className="top-move-pct">
                  {Math.round(m.weight * 100)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="status-line">{data.status}</div>
    </div>
  );
}
