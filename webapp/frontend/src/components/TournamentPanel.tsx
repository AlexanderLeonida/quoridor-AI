import { useWebSocketStream } from "../useWebSocket";
import type { TournamentState } from "../types";
import { Board } from "./Board";
import { HeadToHeadMatrix } from "./HeadToHeadMatrix";

interface Props {
  /** Dense = used in the multi-column dashboard. Stacks board + standings
   *  vertically instead of side-by-side. */
  dense?: boolean;
}

export function TournamentPanel({ dense }: Props) {
  const { data, status } = useWebSocketStream<TournamentState>("tournament");

  if (!data) {
    return (
      <div className={`panel ${dense ? "panel-dense" : ""}`}>
        <div className="panel-header">
          <h2>Round-Robin Tournament</h2>
          <span className={`pill pill-${status}`}>{status}</span>
        </div>
        <div className="panel-empty">waiting for data…</div>
      </div>
    );
  }

  const ranked = [...data.standings]
    .map((s, idx) => ({ ...s, idx }))
    .sort((a, b) => b.rating - a.rating || b.wins - a.wins);

  const a0Name =
    data.match && data.agents[data.match[0]]
      ? data.agents[data.match[0]].name
      : "—";
  const a1Name =
    data.match && data.agents[data.match[1]]
      ? data.agents[data.match[1]].name
      : "—";

  return (
    <div className={`panel tournament-panel ${dense ? "panel-dense" : ""}`}>
      <div className="panel-header">
        <div className="panel-title">
          <h2>Round-Robin Tournament</h2>
          <div className="panel-subtitle">
            5 real iterations from the training run · ratings update live
          </div>
        </div>
        <span className={`pill pill-${status}`}>
          {data.games_completed} games
        </span>
      </div>

      <div className={`tournament-body ${dense ? "tournament-body-dense" : ""}`}>
        <div className="tournament-left">
          <div className="match-label">
            <span className="match-side match-side-p1">{a0Name}</span>
            <span className="vs">vs</span>
            <span className="match-side match-side-p2">{a1Name}</span>
          </div>
          {data.board ? (
            <div className="board-wrap">
              <Board state={data.board} />
            </div>
          ) : (
            <div className="panel-empty">no game active</div>
          )}
        </div>

        <div className="tournament-right">
          <table className="standings">
            <thead>
              <tr>
                <th>#</th>
                <th>Name</th>
                <th>W</th>
                <th>L</th>
                <th>D</th>
                <th>Rating</th>
              </tr>
            </thead>
            <tbody>
              {ranked.map((row, i) => (
                <tr key={row.name}>
                  <td>{i + 1}</td>
                  <td className="name-cell">{row.name}</td>
                  <td>{row.wins}</td>
                  <td>{row.losses}</td>
                  <td>{row.draws}</td>
                  <td className="rating-cell">{Math.round(row.rating)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="status-line">{data.message}</div>
        </div>

        {!dense && data.head_to_head && (
          <div className="tournament-h2h">
            <HeadToHeadMatrix
              agentNames={data.agents.map((a) => a.name)}
              matrix={data.head_to_head}
            />
          </div>
        )}
      </div>
    </div>
  );
}
