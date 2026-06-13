import type { HeadToHeadCell } from "../types";

interface Props {
  agentNames: string[];
  matrix: HeadToHeadCell[][];
}

function colorFor(rate: number, alpha: number): string {
  // Diverging palette: 0 (red) → 0.5 (neutral) → 1 (green)
  if (rate < 0.5) {
    const t = (0.5 - rate) * 2;
    return `rgba(231, 76, 60, ${0.12 + t * 0.55})`;
  }
  const t = (rate - 0.5) * 2;
  return `rgba(46, 204, 113, ${0.12 + t * 0.55})`;
}

/**
 * Pairwise win-rate heatmap.  Rows are "row agent as side-0", columns are
 * "column agent as opponent".  Cell text shows `W-L(-D)` from row's view;
 * background tint encodes row's win rate (green wins, red losses).
 */
export function HeadToHeadMatrix({ agentNames, matrix }: Props) {
  const n = agentNames.length;
  if (n === 0 || matrix.length === 0) return null;

  return (
    <div className="h2h-wrap">
      <div className="h2h-title">
        Head-to-Head — <span className="h2h-axis">row vs column</span>
      </div>
      <div className="h2h-grid-wrap">
        <table className="h2h-grid">
          <thead>
            <tr>
              <th className="h2h-corner" />
              {agentNames.map((name) => (
                <th key={`col-${name}`} className="h2h-colhead">
                  <span>{name}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {agentNames.map((rowName, i) => (
              <tr key={`row-${rowName}`}>
                <th className="h2h-rowhead">{rowName}</th>
                {agentNames.map((_, j) => {
                  if (i === j) {
                    return (
                      <td key={`d-${i}-${j}`} className="h2h-cell h2h-diag">
                        —
                      </td>
                    );
                  }
                  const cell = matrix[i]?.[j] ?? {
                    wins: 0,
                    losses: 0,
                    draws: 0,
                  };
                  const total = cell.wins + cell.losses + cell.draws;
                  if (total === 0) {
                    return (
                      <td
                        key={`e-${i}-${j}`}
                        className="h2h-cell h2h-empty"
                      >
                        ·
                      </td>
                    );
                  }
                  const rate = (cell.wins + 0.5 * cell.draws) / total;
                  return (
                    <td
                      key={`c-${i}-${j}`}
                      className="h2h-cell"
                      style={{ background: colorFor(rate, 0.6) }}
                      title={`${rowName} vs ${agentNames[j]}: ${cell.wins}W ${cell.losses}L ${cell.draws}D`}
                    >
                      <span className="h2h-cell-record">
                        {cell.wins}-{cell.losses}
                        {cell.draws > 0 ? `-${cell.draws}` : ""}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
