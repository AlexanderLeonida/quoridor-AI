import type { BoardState } from "../types";

interface Props {
  state: BoardState;
  size?: number;
  highlightWinner?: boolean;
}

const BOARD_SIZE = 9;
const WALL_GRID = 8;

const COLOR_BG = "#0f1018";
const COLOR_CELL = "#3a3a4d";
const COLOR_GOAL_P1 = "#5a2a2a";
const COLOR_GOAL_P2 = "#2a3a5a";
const COLOR_WALL = "#d4a574";
const COLOR_P1 = "#e74c3c";
const COLOR_P2 = "#3498db";

export function Board({ state, size = 440, highlightWinner = true }: Props) {
  const margin = size * 0.045;
  const inner = size - 2 * margin;
  const cellPlusGap = inner / BOARD_SIZE;
  const wallThickness = Math.max(3, cellPlusGap * 0.16);
  const cell = cellPlusGap - wallThickness;

  const visualRow = (r: number) => BOARD_SIZE - 1 - r;
  const wallVisualR = (r: number) => WALL_GRID - 1 - r;

  const cellXY = (r: number, c: number) => ({
    x: margin + c * cellPlusGap,
    y: margin + visualRow(r) * cellPlusGap,
  });

  const cells: React.ReactNode[] = [];
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const { x, y } = cellXY(r, c);
      let fill = COLOR_CELL;
      if (r === 8) fill = COLOR_GOAL_P1;
      else if (r === 0) fill = COLOR_GOAL_P2;
      cells.push(
        <rect
          key={`c-${r}-${c}`}
          x={x}
          y={y}
          width={cell}
          height={cell}
          fill={fill}
          rx={3}
        />,
      );
    }
  }

  const walls: React.ReactNode[] = [];
  for (const [r, c] of state.h_walls) {
    const vr = wallVisualR(r);
    const x = margin + c * cellPlusGap;
    const w = 2 * cellPlusGap - wallThickness;
    const y = margin + (vr + 1) * cellPlusGap - wallThickness;
    walls.push(
      <rect
        key={`h-${r}-${c}`}
        x={x}
        y={y}
        width={w}
        height={wallThickness}
        fill={COLOR_WALL}
        rx={1.5}
      />,
    );
  }
  for (const [r, c] of state.v_walls) {
    const vr = wallVisualR(r);
    const x = margin + (c + 1) * cellPlusGap - wallThickness;
    const y = margin + vr * cellPlusGap;
    const h = 2 * cellPlusGap - wallThickness;
    walls.push(
      <rect
        key={`v-${r}-${c}`}
        x={x}
        y={y}
        width={wallThickness}
        height={h}
        fill={COLOR_WALL}
        rx={1.5}
      />,
    );
  }

  const pawns = state.pawns.map(([r, c], idx) => {
    const { x, y } = cellXY(r, c);
    const color = idx === 0 ? COLOR_P1 : COLOR_P2;
    const winning = highlightWinner && state.winner === idx;
    return (
      <g key={`p-${idx}`}>
        {winning && (
          <circle
            cx={x + cell / 2}
            cy={y + cell / 2}
            r={cell * 0.45}
            fill="none"
            stroke="#f1c40f"
            strokeWidth={3}
            opacity={0.85}
          />
        )}
        <circle
          cx={x + cell / 2}
          cy={y + cell / 2}
          r={cell * 0.32}
          fill={color}
          stroke="white"
          strokeWidth={2}
        />
      </g>
    );
  });

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      width="100%"
      preserveAspectRatio="xMidYMid meet"
      style={{ background: COLOR_BG, borderRadius: 10, display: "block" }}
    >
      {cells}
      {walls}
      {pawns}
    </svg>
  );
}
