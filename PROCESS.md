# Process

How the Quoridor engine evolved from a single-file alpha-beta program into a self-playing neural net trained under a calibrated Elo regime. For each step: *what* we changed and *why*.

---

## 1. Classical search baseline

**Commits `a972739` → `abb4d31` → `15c517d`.**

- `Board` class with rules, legal-move generation, BFS for reachability checks (wall placement must not fully block either player's goal).
- **Negamax with principal variation search (PVS)** + iterative deepening under a time budget.
- **Zobrist hashing** for a 64-bit board key → transposition table with EXACT / LOWER / UPPER bound tags and a TT-best-move-first move ordering heuristic.
- **Killer moves** (two slots per ply).
- **Wall pruning**: only consider wall anchors adjacent to a cell on either player's current shortest path. Standard Quoridor heuristic — drops the effective branching factor by ~10x without meaningfully changing play strength.
- Evaluation from side-to-move: `(opp_path − my_path) · 100 + wall_diff · 6 + mobility_diff · 2 + advance_diff + tempo`.
- **Terminal scores decay with ply** (`WIN_SCORE − ply`) so the engine prefers shorter wins and longer losses.
- CLI + Tkinter GUI with difficulty levels (Easy: 2s/depth 3, Medium: 8s, Hard: 30s).

**Why:** Getting the engine fast at depth 4-6 gives us a reference opponent (and a way to test the rules engine against a human). Later this is the baseline the NN has to beat.

---

## 2. Database + canonical encoding

**Before the NN.** Built the training-data substrate:

- `quoridor/database.py` — SQLite with `games(id, winner, plies, p1_source, p2_source, model_version, notes, ...)` and `moves(game_id, ply, side, kind, r, c, elapsed_ms, policy_blob)`. Stores only move lists, not tensors — positions are re-materialised on demand by replaying. Makes the DB ~40× smaller and robust to encoding changes.
- `quoridor/encoding.py` — defines the contract between DB and network:
  - **Input planes (7, 9, 9)**: my pawn, opp pawn, h-walls, v-walls, my walls-left / 10, opp walls-left / 10, all-ones bias.
  - **Canonical view**: the network always sees the position from side-to-move's perspective as P0 (row-flipped when real side-to-move is P2). A `flipped` flag is threaded through `move_to_action` / `action_to_move` so moves are mapped into the same canonical frame.
  - **Action space (209)**: 81 pawn-cells + 64 horizontal wall anchors + 64 vertical wall anchors.
  - `serialize_policy` / `deserialize_policy` for storing MCTS visit distributions as BLOBs in the DB.

**Why:** The 40× savings matters — even after 14k+ self-play games the DB stays small. Canonical view halves the learning problem (net only sees one "perspective"). Storing raw tensors in the DB would have locked us to a single encoding version; storing moves lets us change encoding and re-materialise.

---

## 3. MCTS + the stalling problem

**Commit `9adc041`.**

Alpha-beta on Quoridor hits a *safe-stalling equilibrium*: two even players can each place one more wall and still be equal, so a purely minimax evaluator picks waiting moves. Added AlphaZero-style MCTS (`quoridor/mcts.py`):

- **PUCT exploration** with log-scaling `c_puct` (MuZero/KataGo formula).
- **Dirichlet noise** at root priors (α=0.3, ε=0.25) for exploration diversity in self-play.
- **First Play Urgency (FPU)** reduction on unvisited children.
- **Temperature schedule**: proportional sampling on visit counts for the first ~15 full moves, greedy afterwards.
- **Correct negamax value propagation** for the two-player zero-sum game.
- **EvalCache**: bounded (20k entries) map keyed on Zobrist hash → (policy_logits, value). Quoridor has real transpositions (pawn-then-wall vs wall-then-pawn), so this skips NN forwards on repeats.

**Why:** Visit counts at the root encode a policy that commits to advancing rather than parking on wall placements. MCTS also gives us soft policy targets (visit distributions) that are richer training signal than a bare move.

---

## 4. Neural net + AlphaZero-style self-play RL

Added `quoridor/net.py` (ResNet) and `selfplay.py` (pipeline):

- Starting architecture: **6 blocks × 64 filters**, 3×3 conv stem → residual tower → policy head (1×1 conv → FC → 209 logits) → value head (1×1 conv → FC(64) → FC(1) → tanh).
- `selfplay.py` implements the loop:
  1. **Generate** games with MCTS (visit distributions stored as soft policy targets via `policy_blob`).
  2. **Train** — CE(policy soft targets) + MSE(value) + L2, cosine LR annealing, gradient clipping.
  3. **Gate**: new-net-vs-old head-to-head; promote only if score > 0.52.
  4. Repeat.
- PyTorch imported lazily so engine/DB work on a torch-less machine.

**Why 6×64 first:** Small enough to train tractably on Apple Silicon MPS. Enough capacity to learn wall tactics. Only scale up after confirming the full pipeline works.

---

## 5. Draw handling

Quoridor has no formal draw, but at a max-moves cap games stall. Initial treatment: draws were 0 value for both sides. Problem: the "safe stalling" equilibrium reappeared — the net learned that drawing was fine.

Fixes added:
- **Draw penalty** (`--draw-penalty`, default 0.1 initially, later raised to 0.5): drawn games get `z = -draw_penalty` so both sides are punished for drawing.
- **Progress-aware draw values**: shortest-path difference at the final board modulates the penalty — the side that was closer to winning gets less negative z.
- **Stall scaling**: longer draws (closer to the cap) are penalised harder. A 30-ply premature draw is less punished than a 90-ply full-cap stall.

**Why:** Without this, MCTS was happy to drive toward balanced wall positions that ran out the clock.

---

## 6. Adjudication at max-moves

To convert stall draws into decisive signal:
- **`adjudicate_winner(board, min_gap)`**: at the move cap, the side with the shorter shortest-path wins iff the gap is `≥ min_gap`.
- Defaults evolved: `adjudicate_gap=2` (conservative) → later tightened to `1` (aggressive) as we needed more decisive eval.

Later (current session) we discovered that **eval didn't use adjudication at all** — only self-play. Games that timed out in eval were always draws, which inflated the eval draw rate to 70%+. Fixed by threading `adjudicate_gap` into `evaluate_nets` / `evaluate_nets_parallel` / the eval worker.

**Why:** Adjudication converts ~50% of drawn games into decisive outcomes in training, and uniformly applying it between self-play and eval makes the two contracts consistent.

---

## 7. Parallel self-play + NN eval cache

**Commit `26f3920`.**

Self-play dominated wall-clock, and games are independent:
- `--workers N` spawns N worker processes (spawn context so PyTorch + forking don't fight).
- Each worker loads the net to CPU once (from a `_worker_net.pt` snapshot) — CPU is fine for a tiny net, and avoids cross-process MPS sharing complexity.
- Workers return `(moves, policy_blobs, winner, plies, elapsed)`; main process does all DB writes (SQLite + multiprocess don't mix).
- `torch.set_num_threads(1)` in each worker prevents thread oversubscription.

Shift: ~10 games/hr → ~100+ games/hr on a 12-core Mac.

---

## 8. Overfitting diagnosis + larger replay buffer

**Commit `ed69375`** ("after 60k games, model is clearly overfitting").

Training loss kept falling while gating win-rate flattened. Buffer was too small (~1k games), so each epoch saw the same narrow distribution.

- Replay buffer bumped to **50k** games (later normalised to `--window`, typically 3000-5000 at a time).
- Eval games raised 20 → 50 to reduce gate noise.
- **Game-level train/val split** (not position-level) so positions from the same game never leak between train and val.
- **Per-position sample weight**: inversely proportional to game length, with a 4× decisive-multiplier on games with a real winner — short decisive games weren't getting drowned out by long stall-draws.
- **Auto-resume** from `best.pt` if present (we kept forgetting `--resume`).
- **Best-val-weights snapshot**: during each training call, keep the epoch with the lowest val loss and restore those weights at the end — so the candidate we ship is the best point we saw, not the final (typically overfit) one.

---

## 9. Parallel evaluation + tree reuse

**Commit `484ac13`.**

At 50 eval games the serial evaluator was eating most of each iteration's wall-clock. Games between two fixed nets are independent, so we parallelised eval identical to self-play: workers load both nets once, play independently, results stream back via `imap_unordered`.

Separate optimisation in the same commit: **subtree reuse** in MCTS. Instead of throwing away the tree after each move, keep the chosen child's expanded subtree as the next search's root. 30-50% of the NN evaluations already invested under that node carry over as "free" simulations.

---

## 10. Early termination in MCTS

Added: every 16 simulations, check whether the leading child's visit count is unreachable by the runner-up even if all remaining sims go to it. If so, break early — the action selection is already decided.

**Why:** In one-sided positions the search was wasting 200+ sims confirming an obvious first choice.

---

## 11. Statistical gating: CI + WLD

**Commit `cc4481c`.**

Gating at 50 games had ~14pp standard error. We were promoting on noise.

- Eval games raised 50 → 200 (later 30 in iteration-constrained settings).
- **Wilson-score 95% CIs** reported alongside the raw percentage.
- **Per-net W/L/D printed separately** so drift toward draws was visible (e.g. "16W/14L/0D" vs "5W/17L/8D" look like different kinds of 52%).

**Gate threshold evolution:**
- Initially 0.55 (required a clear win).
- Dropped to 0.52 when 200-game CIs proved we could trust smaller signals.
- Briefly considered 0.50 + `--no-gate` runs when trying to unblock a plateau, but rejected because gate protection is critical.

---

## 12. Policy-target sharpening + value-weight hyperparameter

Added `--policy-temp` (default 0.7, `<1` sharpens MCTS visit-count targets via `counts ** (1/temp)`). Low-sim positions have flatter visit distributions; sharpening focuses learning on the peaks.

Added `--value-weight` so we can down-weight the MSE value loss relative to the policy CE. After distillation (next section), a well-calibrated value head shouldn't be pushed around by blunt outcome targets — we used `--value-weight 0.1-0.3` for a few runs.

---

## 13. Eval-time temperature

Between near-identical nets, greedy play produces deterministic mirror games that end in draws. Added `--eval-temp` / `--eval-temp-moves` — for the first N plies of eval, sample from visit counts with a small temperature (e.g., 0.5) before going greedy. Enough to break mirror lines while keeping the eval signal honest.

---

## 14. Output-bug fix

**Commit `e3b5853`.** A stray mislabeling in terminal output was confusing our progress tracking. Cosmetic but blocking diagnosis.

---

## 15. Architecture upgrade: 6×64 → 10×128 + distillation

**Commit `a3f40df`.**

6×64 had clearly plateaued — policy loss bottomed out around 1.3, Elo climbed then stagnated at ~1030 peak (v74). A 10×128 net has enough capacity for deeper wall tactics and longer-range path planning, but training from scratch would throw away all the knowledge in v74.

**Distillation (`distill.py`):**
- Sample ~80k positions from the v2 DB.
- Forward each through v74 (teacher) to get soft policy + value targets.
- Train the 10×128 student to match those targets.
- After distillation, bench showed distilled-10×128 ≈ 48% vs v74 → essentially at parity, ready for self-play to push past.

The 10×128 build is `build_net(blocks=10, filters=128)`; checkpoint configs persist the shape so `load_checkpoint` reconstructs the right architecture for old 6×64 files.

---

## 16. Database evolution

`data/quoridor.db` → `quoridor_v2.db` → `quoridor_v3.db`.

- `quoridor.db`: original, mixed early-era data.
- `quoridor_v2.db`: the 6×64 self-play era — ~14k games across v1-v82.
- `quoridor_v3.db`: clean start for the 10×128 architecture after distillation — keeps the old 6×64 games separate so training on recent self-play doesn't mix architectural distributions.

---

## 17. Current session: column-flip data augmentation

Quoridor is symmetric about the central column: any position has a valid column-mirror with the same value and a permuted policy.

Added to `quoridor/encoding.py`:
- `COL_FLIP_PERM` — 209-element int array, the action permutation under column flip.
- `col_flip_state` — flips the (7, 9, 9) tensor along its last dim.
- `col_flip_policy` — permutes a 209-vector.

In `selfplay.train_on_recent_games`: concat augmented copy with original, so each epoch sees 2× unique samples. On small training windows this halves the train/val overfit gap at zero compute cost.

---

## 18. LR warmup + cosine

Replaced plain `CosineAnnealingLR` with a `LambdaLR`: linear warmup for first 5% of steps, then cosine anneal to zero. Stabilises early-epoch loss on low-data iterations and prevents the first few batches from overshooting.

---

## 19. Eval-time adjudication (discovered bug)

As noted in §6: until this session, `evaluate_nets` / `evaluate_nets_parallel` treated all max-moves cutoffs as draws. Observed symptom: one eval batch had 21 draws out of 30 games — gating signal was mostly noise.

Fix: plumbed `adjudicate_gap` through eval. Same contract as self-play. Draw rate collapsed from ~70% to typical single-digit percentages.

---

## 20. Round-robin tournament + global Elo calibration

**Commit `029d283`.**

**Symptom:** After a long chain of gate-passing promotions, a full round-robin revealed the current `best.pt` (v52) was 9th of 10. Even the pre-self-play warmstart (10×128 before any self-play) was beating later promoted versions.

**Root cause:** Gating Elo is a **local** signal.
- K=32 per head-to-head; one promotion gains ~5 Elo.
- A chain of candidates that each beat their predecessor 52-55% can collectively drift *downward* — no single gate match ever compares to anything older than the previous best.
- Stale models (v74 at Elo 1032) never play again → their rating is frozen from a past era with weaker opponents.

**Fix — `tournament.py`:**
- Round-robin among a specified list of checkpoints.
- **Bradley–Terry MLE** on all match outcomes solves for globally-consistent Elos (anchored to a user-chosen label, default 1000).
- `--save-to-db` / `--save-champion-only` writes the Elo champion's winning games (with MCTS visit policies) into the self-play DB — they become high-quality training signal for the next training pass.

---

## 21. Automated in-training tournaments (revert-to-champion)

`selfplay.py` now invokes `tournament.py` as a subprocess every `--tournament-every N` iterations:

- **Pool** = rolling buffer of last K promoted nets + held-out anchors (`--tournament-anchors`, repeatable).
- Run round-robin, compute MLE Elos.
- If the champion is not the current `best_net`, **revert** — load the champion's weights into `best_net` and re-save `best.pt`.
- Champion games are already in the DB, so the next training pass distils from them.

This changes the selection regime from **gating only** to a **hybrid**:
- Per-iteration gate for fast iteration and preventing single-step disasters.
- Every-N-iteration round-robin for drift correction across chains.

**Gotcha caught during this session:** Every iteration saves `best_net` to `iter_{global_it:04d}.pt`, overwriting whatever was there before. Running twice over the same global iteration range (e.g. after a restart) clobbers historical checkpoints. So after the tournament told us v36 was best, I rolled back `best.pt` to `iter_0036.pt` — but subsequent iterations then overwrote `iter_0036.pt` with v34 weights. Noted: needs a rename/copy step before continuing, or save candidates to a separate filename.

---

## Current state

- Net: 10 blocks × 128 filters, distilled from v74 (6×64 peak).
- `best.pt` currently set to `iter_0034.pt` weights (the tournament-calibrated strongest in the first round-robin).
- A second round-robin suggests `warmstart_10x128.pt` (pre-self-play distillation) is among the strongest — **every self-play-trained net is at-or-below warmstart** in the latest data, meaning self-play has been net-negative since distillation.
- Next step under consideration: restart from warmstart with much gentler optimiser settings (`lr 1e-4`, `epochs 1`, stronger L2) so self-play refines rather than erodes the distilled knowledge.
