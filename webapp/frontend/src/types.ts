export interface BoardState {
  pawns: [number, number][];
  h_walls: [number, number][];
  v_walls: [number, number][];
  walls_left: [number, number];
  turn: number;
  winner: number | null;
}

export interface TopMove {
  notation: string;
  visits: number;
  weight: number;
  value: number;
}

export interface SpectatorState {
  title: string;
  p1_name: string;
  p2_name: string;
  board: BoardState;
  value: number;
  last_move: string | null;
  score: [number, number, number];
  game_num: number;
  status: string;
  history: string[];
  top_moves: TopMove[];
}

export interface ProjectStats {
  games_total?: number;
  moves_total?: number;
  iterations_max?: number;
  iterations_logged?: number;
  promotions?: number;
  latest_train_loss?: number;
  arch_blocks?: number;
  arch_filters?: number;
  param_count?: number;
  checkpoints_saved?: number;
}

export interface StandingRow {
  name: string;
  wins: number;
  losses: number;
  draws: number;
  rating: number;
}

export interface HeadToHeadCell {
  wins: number;
  losses: number;
  draws: number;
}

export interface TournamentState {
  board: BoardState | null;
  match: [number, number] | null;
  agents: { name: string }[];
  standings: StandingRow[];
  head_to_head: HeadToHeadCell[][];
  message: string;
  games_completed: number;
}

export interface MetricsRow {
  global_iter?: number;
  train_loss?: number;
  policy_loss?: number;
  value_loss?: number;
  sp_p1_wins?: number;
  sp_p2_wins?: number;
  sp_draws?: number;
  sp_avg_plies?: number;
  eval_score?: number;
  promoted?: number;
  [k: string]: number | string | null | undefined;
}

export interface MetricsState {
  rows: MetricsRow[];
}
