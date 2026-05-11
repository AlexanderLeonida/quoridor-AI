# Quoridor-AI — Resume Bullets (FAANG-targeted, XYZ format)

10 strongest 2–3 bullet groupings. Each bullet leads with a quantitative metric, follows the "Accomplished [X] by [Y] measured by [Z]" structure, and names the concrete tech stack for ATS/recruiter parsing.

---

## Final MLE Infra 

Quoridor-AI | Distributed RL Training System
- Scaled distributed self-play 10× (10 → 100+ games/hour) by building a multiprocessing training pipeline with 30-50% MCTS inference speedup for the strategy board game Quoridor
- Trained a PyTorch policy-value neural network beating alpha-beta search at depth 8 across Quoridor’s 200+ action space by combining self-play with search-guided distillation for strong real-time gameplay

## Alex Changed

**MLE**

Quoridor-AI | Reinforcement Learning System

- Scaled distributed self-play 10× (10 → 100+ games/hour) by building a multiprocessing training pipeline with 30-50% MCTS inference speedup for the strategy board game Quoridor
- Trained a PyTorch policy-value neural network beating alpha-beta search at depth 8 across Quoridor's 200+ action space by combining self-play with search-guided distillation for strong real-time gameplay

**SWE**

Quoridor-AI | High-Performance AI System
• Built a high-performance AI system in Python improving decision performance 8× through neural-guided search and iterative optimization techniques
• Reduced compute latency 30–50% via caching, subtree reuse, and parallelized search pruning across a 200+ action decision space
• Increased system throughput 10× (10 → 100+ simulations/hour) by engineering a multiprocessing architecture with shared replay storage and coordinated worker execution

### AlphaZero-Style MCTS Engine | PyTorch, NumPy, Zobrist Hashing

- Built an AlphaZero-style PUCT search engine in PyTorch reaching +893 Elo over a randomly-initialized baseline by integrating Dirichlet root noise (α=0.3, ε=0.25), First-Play Urgency reduction, MuZero log-scaled c_puct, and negamax value backup over a 209-action policy space
- Cut neural net forward passes 30–50% per move by implementing subtree reuse across MCTS calls and a 20k-entry Zobrist-keyed transposition cache for repeated positions arising from Quoridor's pawn/wall move-order symmetry
- Eliminated 200+ wasted simulations per move by adding a mathematical early-termination check every 16 sims that breaks when the leading child's visit count becomes provably unreachable by the runner-up

### Negamax + PVS Alpha-Beta Engine | Python, Bitwise Zobrist, Iterative Deepening

- Built a 30-second-budget iterative-deepening **Principal Variation Search** engine reaching depth 8 on 9×9 Quoridor by combining 64-bit Zobrist transposition tables (EXACT/LOWER/UPPER bounds), two-slot killer-move heuristics per ply, and ply-decayed terminal scoring (`WIN_SCORE − ply`) to prefer shorter wins
- Reduced effective branching factor ~10× by implementing shortest-path-anchored wall pruning (BFS-based — only consider wall anchors adjacent to either player's current shortest path) without measurable loss in playing strength

### Knowledge Distillation Pipeline | PyTorch, KL-Regularized Continual Learning

- Achieved a **+366 Elo** single-shot improvement on a 10×128 ResNet by distilling a depth-8 alpha-beta teacher onto 3000 sampled board positions, exporting soft policy + value targets and training student over CE + MSE loss
- Mitigated catastrophic forgetting across 5 successive distillation rounds (cumulative +893 Elo) by adding a `λ·KL(student ‖ pre_distill_student)` regularizer (λ=0.5) plus 30% replay-buffer rehearsal from prior self-play data, preserving capability on the prior distribution while absorbing the new teacher

### Parallel Self-Play RL Pipeline | Python multiprocessing, PyTorch, SQLite

- Scaled self-play game throughput **10×** (10 → 100+ games/hour) by spawning 8 worker processes via Python `spawn` context, snapshotting weights to a `_worker_net.pt` file, pinning `torch.set_num_threads(1)` per worker to avoid oversubscription, and centralizing all SQLite writes on the main process
- Broke the closed-loop self-play drift attractor by injecting 20% NN-vs-alpha-beta games into each iteration's data — produced the project's first 6 net promotions in 16 iterations after 12 prior pure-self-play iterations failed to beat the warmstart
- Cut training-data overfit gap ~50% at zero compute cost by adding column-flip data augmentation (`COL_FLIP_PERM` 209-element action permutation + tensor mirror), exploiting Quoridor's central-column symmetry

### Bradley-Terry Tournament Calibration | NumPy MLE, Bootstrap CIs

- Diagnosed gating-Elo drift (current `best.pt` ranked **9th of 10** in round-robin tournament) and built a Bradley-Terry MLE solver with anchored ratings + bootstrap 95% confidence intervals to compute globally-consistent Elos across versioned checkpoints
- Prevented spurious tournament reverts (one near-loss of a +366-Elo distillation gain to dict-ordering noise on tied 2-2 results) by introducing a 25-Elo revert-gap threshold and doubling per-pair sample size 4 → 8, halving sampling variance
- Eliminated duplicate-checkpoint pollution in tournament pools by hashing the stem-conv layer of every loaded net to fingerprint and dedupe rolled-back identical weights

### Hard-Example Mining + Lightweight PBT | PyTorch, In-Memory Replay

- Recovered tournament-champion knowledge by mining **861 positions** per revert event where the rejected candidate disagreed with the champion's MCTS top move, passed in-memory to the next training pass at 8× normal sample weight — produced the first net to ever surpass the long-standing v34 baseline at 56.7% gating
- Explored hyperparameter neighborhoods at 2× compute cost by spawning sibling candidates every 6 iterations with mutated learning rate (×0.5/×1.5/×2.0) and weight decay (×0.5/×2.0), keeping whichever achieved lower validation loss

### Auxiliary Value Supervision + Calibrated Draw Handling | PyTorch

- Collapsed value-head MSE loss **5×** (0.5 → 0.1) by blending sparse outcome targets `z ∈ {-1,0,+1}` with a dense per-position `tanh((opp_path − my_path) / 6)` board-geometry signal at α=0.4, densifying value supervision ~30× per game
- Eliminated the safe-stalling equilibrium attractor by replacing zero-value draws with a stall-scaled penalty `−(0.5 + 0.4·plies/max_moves) + 0.5·tanh(path_diff/6)`, converting stalled draws into graded signal proportional to ply length and final shortest-path gap
- Converted ~50% of stalled max-move games into decisive training signal by implementing `adjudicate_winner(board, min_gap)` — awards the win to the side with the shorter shortest-path when the gap exceeds threshold, plumbed identically through self-play and eval (fixed an inflated 70%+ eval draw rate)

### Statistical Gating + Wilson Confidence Intervals | NumPy, scipy.stats

- Reduced gating false-promotion rate by **4×** by raising evaluation game count 50 → 200, reporting Wilson-score 95% CIs alongside raw win percentages, and surfacing per-net W/L/D breakdowns to expose latent draw-drift hidden in headline 52% scores (e.g., distinguishing 16W/14L/0D from 5W/17L/8D)
- Eliminated deterministic mirror-game draws between near-identical candidate nets by adding eval-time temperature sampling (τ=0.5 over the first 10 plies) before reverting to greedy play, preserving signal honesty while breaking visit-count ties

### Game Database + Canonical Encoding | SQLite, NumPy

- Reduced training-database storage **40×** across 14k+ self-play games by persisting only move-lists in SQLite (not tensors) and re-materializing positions on demand by replaying from `Board.initial()`, decoupling the on-disk format from neural-net encoding evolution
- Halved the learning problem size by enforcing a canonical side-to-move tensor view across 7×9×9 input planes (own pawn / opp pawn / h-walls / v-walls / own walls-left / opp walls-left / bias) with a 209-action policy head (81 pawn cells + 64 horizontal + 64 vertical anchors), threading a `flipped` flag through `move_to_action`/`action_to_move` so policy targets stay in the canonical frame

### Adversarial Diagnostic Suite + Human-Game Mining | PyTorch, Custom Search Oracles

- Diagnosed a value-head racing bias by building a 12-game forward-rusher diagnostic that revealed the trained NN placed walls with **0% useful delay** vs depth-3 alpha-beta's 82%, isolating the failure to the `tanh(path_diff/6)` aux-value bias penalizing turn-cost defensive walls
- Mined **24 adversarial supervision positions** from 4 instrumented human-vs-NN games (Tkinter GUI hooked into `WM_DELETE_WINDOW` for safe game-end recording) where the bot missed walls with up to +11 path-delay value, materialized as oracle (state, wall_action, value=+1) training triples
- Validated targeted distillation effectiveness by lifting oracle-wall top-1 prediction from **4% → 71%** on the mined positions via 8-epoch weighted training (10× sample weight + KL anchor at λ=0.5), confirmed via held-out forward-rusher evaluation that core racing capability was preserved (6-0 sweep)
