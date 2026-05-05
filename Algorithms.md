# Algorithms

This document describes the core algorithms used to train and evaluate
the Quoridor AI. Each section gives pseudocode followed by exact
references into the codebase.

---

## 1. Monte Carlo Tree Search (PUCT, AlphaZero-style)

Tree search guided by the policy/value network. Visit-count distributions
become the policy targets used to train the next iteration.

### Pseudocode

```
function MCTS(board, net, config):
    root = new Node()
    p_logits, v = net(encode(board))
    expand(root, board, p_logits, v)        # create child nodes with priors
    add_dirichlet_noise(root)               # exploration at the root only

    repeat config.num_simulations times:
        node    = root
        scratch = board
        path    = [root]

        # SELECT — descend along PUCT-best children until a leaf
        while node.expanded and not node.terminal:
            action, child = argmax_a UCB(node, child_a)
            scratch = scratch.apply(action_to_move(action))
            path.append(child)
            node = child

        # EXPAND + EVALUATE
        if node.terminal:
            leaf_value = node.terminal_value
        else:
            p_logits, v = net(encode(scratch))     # cache by Zobrist
            leaf_value  = expand(node, scratch, p_logits, v)

        # BACKUP — propagate value with negamax sign-flips
        backup(path, leaf_value, node.to_play)

    return root
```

```
function UCB(parent, child):
    pb_c = log((parent.N + c_base + 1) / c_base) + c_init           # MuZero scaling
    prior_score = pb_c * child.prior * sqrt(parent.N) / (1 + child.N)
    if child.N > 0:
        value_score = -child.Q                                       # negamax flip
    else:
        value_score = parent.Q - fpu_reduction                       # First Play Urgency
    return value_score + prior_score
```

```
function expand(node, board, p_logits, v):
    if board.winner() exists:
        node.terminal       = true
        node.terminal_value = ±1 from to_play's POV
        return node.terminal_value
    mask                  = legal_action_mask(board)
    priors                = softmax(p_logits where mask else -inf)
    for each legal action a:
        node.children[a]  = new Node(prior = priors[a])
    return v

function backup(path, value, leaf_to_play):
    for node in reverse(path):
        sign           = +1 if node.to_play == leaf_to_play else -1
        node.value_sum += sign * value
        node.visit_count += 1
```

Optional optimisations layered into `search()`:

- **Subtree reuse**: keep the chosen child as the next move's root so
  visits from the previous search are inherited.
- **Eval cache**: Zobrist-keyed transposition table for NN forwards.
- **Early termination**: stop simulating once the leading child's visit
  count is mathematically unreachable by the runner-up.

After `search()` finishes, the policy target is built from visit counts:

```
function get_policy(root, temperature):
    counts = [child.N for child in root.children]
    if temperature == 0:
        return one-hot at argmax(counts)
    return (counts ** (1/temperature)) / sum
```

### Source references

- `quoridor/mcts.py:113` — `Node` dataclass with visit/value/prior fields.
- `quoridor/mcts.py:157` — `_ucb_score()` (PUCT formula with FPU).
- `quoridor/mcts.py:178` — `_select_child()` (argmax descent).
- `quoridor/mcts.py:193` — `_add_dirichlet_noise()` (root exploration).
- `quoridor/mcts.py:205` — `_expand()` (masked-softmax priors, terminal handling).
- `quoridor/mcts.py:247` — `_evaluate()` (NN forward + Zobrist cache lookup).
- `quoridor/mcts.py:276` — `_backup()` (negamax value propagation).
- `quoridor/mcts.py:291` — `search()` (public entry — fresh root, subtree reuse, early-termination loop at lines 360–377).
- `quoridor/mcts.py:407` — `get_policy()` (visit-count → distribution).
- `quoridor/mcts.py:442` — `select_action()` (sample from policy).
- `quoridor/mcts.py:40` — `EvalCache` (transposition table for NN forwards).
- `quoridor/mcts.py:87` — `MCTSConfig` (hyperparameters: c_base, c_init, dirichlet_alpha/epsilon, fpu_reduction).
- `quoridor/ai.py:91` — `zobrist()` (hash function used by the cache key).

---

## 2. Alpha–Beta (Negamax + PVS)

Classical search with alpha-beta pruning, principal-variation search, a
Zobrist-hashed transposition table, killer-move heuristic, and Quoridor-
specific wall-anchor pruning. Used as a stand-alone strong opponent and
as an alternate teacher for distillation.

### Pseudocode

```
function find_best_move(board, max_depth, time_limit):
    deadline = now() + time_limit
    tt       = empty TranspositionTable
    killers  = [[None, None] for _ in 0..MAX_PLY]
    best     = None
    for depth in 1..max_depth:
        try:
            _, mv = negamax(board, depth, -INF, +INF, ply=0,
                            deadline, tt, killers)
            if mv: best = mv
        except Timeout:
            break
    return best
```

```
function negamax(b, depth, alpha, beta, ply, deadline, tt, killers):
    if now() > deadline: raise Timeout
    if b.winner() is not None:
        return ±(WIN_SCORE - ply), None        # shorter wins beat longer wins
    if depth == 0:
        return evaluate(b), None

    # --- Transposition-table lookup ---
    entry = tt.get(zobrist(b))
    tt_move = None
    if entry and entry.depth >= depth:
        if entry.flag == EXACT: return entry.value, entry.best
        if entry.flag == LOWER: alpha = max(alpha, entry.value)
        if entry.flag == UPPER: beta  = min(beta,  entry.value)
        if alpha >= beta:        return entry.value, entry.best
        tt_move = entry.best

    # --- Move generation + ordering ---
    scored = generate_moves(b)                  # pawn moves + pruned walls
    moves  = order(scored, tt_move, killers[ply])

    best, best_move, first = -INF, moves[0], true
    for m in moves:
        child = b.apply(m)
        if first:
            val = -negamax(child, depth-1, -beta, -alpha, ply+1, ...)
        else:
            # PVS — null-window probe; full re-search only on fail-high
            val = -negamax(child, depth-1, -alpha-1, -alpha, ply+1, ...)
            if alpha < val < beta:
                val = -negamax(child, depth-1, -beta, -val, ply+1, ...)
        first = false

        if val > best: best, best_move = val, m
        if val > alpha: alpha = val
        if alpha >= beta:
            if m is a wall: record m as killer at this ply
            break                                # beta cutoff

    flag = UPPER if best <= alpha_orig else (LOWER if best >= beta else EXACT)
    tt.put(zobrist(b), depth, flag, best, best_move)
    return best, best_move
```

```
function evaluate(b):
    me, opp = b.turn, 1 - b.turn
    if winner exists: return ±WIN_SCORE
    return  (path[opp] - path[me]) * W_PATH
          + (walls[me] - walls[opp]) * W_WALL
          + (mobility[me] - mobility[opp]) * W_MOBILITY
          + (advance[me] - advance[opp]) * W_ADV
          + tempo_bonus
```

Wall pruning is the single biggest speedup: only walls anchored adjacent
to a cell on either player's current shortest path are considered.

### Source references

- `quoridor/ai.py:399` — `find_best_move()` (iterative-deepening driver).
- `quoridor/ai.py:306` — `_negamax()` (PVS / TT / killers).
- `quoridor/ai.py:109` — `evaluate()` (static heuristic).
- `quoridor/ai.py:153` — `_shortest_path_cells()` (BFS for wall pruning).
- `quoridor/ai.py:194` — `_candidate_wall_anchors()` (wall pruning set).
- `quoridor/ai.py:208` — `_generate_moves()` (move gen + ordering scores).
- `quoridor/ai.py:285` — `_order_moves()` (TT-best, killers, then heuristic score).
- `quoridor/ai.py:255` — `_TT` (transposition table with EXACT/LOWER/UPPER flags).
- `quoridor/ai.py:91` — `zobrist()` (hash function).
- `quoridor/ai.py:51` — eval weights (`W_PATH`, `W_WALL`, `W_MOBILITY`, `W_ADV`).

---

## 3. Neural Network Updates (policy + value training)

The net is a residual conv tower with a policy head (logits over the
209-action space) and a value head (scalar in [-1, 1]). Training is
joint cross-entropy + MSE on (state, policy_target, z) triples.

### Pseudocode

```
function train_step(net, batch, optimizer):
    x, π_target, z, w = batch                      # w: per-sample weight
    p_logits, v_pred  = net(x)

    log_p = log_softmax(p_logits)
    L_p   = mean( w * -sum(π_target * log_p, axis=1) )       # soft cross-entropy
    L_v   = mean( w * (v_pred - z) ** 2 )                    # MSE
    loss  = L_p + value_weight * L_v

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(net, max_norm=1.0)
    optimizer.step()
    scheduler.step()                               # warmup → cosine decay
```

The full `train_on_recent_games` pipeline adds:

- **Weighted samples**: shorter / decisive / tournament-champion games
  get higher weight (`decisive_mult * tourney_mult / game_len`).
- **Column-flip augmentation**: the board is symmetric about the centre
  column, so every (state, policy) is mirrored — value unchanged.
- **Aux value blend**: the target z is mixed with a tanh-squashed
  shortest-path differential to give the value head dense supervision.
- **Game-level train/val split** to avoid position leakage.
- **Best-val checkpoint restore**: the candidate weights returned are
  the snapshot with lowest val loss, not the end-of-training weights.
- **LR schedule**: linear warmup over `warmup_frac` of steps, then
  cosine decay to zero.
- **In-memory hard examples**: positions where rejected candidates
  diverged from the champion are appended after a tournament revert.

```
function build_dataset(db, ...):
    for each game in window:
        replay moves; for each (board, move, policy_blob):
            states.append(encode_state(board))
            π = deserialize(policy_blob) or one_hot(move)        # soft if available
            z = ±1 if winner else draw_value(final_board, side, ...)
            weights.append( decisive_mult * tourney_mult / len(game) )
    if augment:
        append column-flipped (state, policy)
    return states, π, z, weights
```

### Source references

- `quoridor/net.py:37` — `build_net()` (residual tower; policy & value heads).
- `quoridor/net.py:54` — `QuoridorNet.forward()` (returns `(p_logits, v)`).
- `quoridor/net.py:100` — `save_checkpoint()` / `load_checkpoint()`.
- `quoridor/encoding.py` — `encode_state()`, `legal_action_mask()`,
  `move_to_action()`, `action_to_move()`, `canonical_view()`,
  `serialize_policy()`, `deserialize_policy()`.

Self-play pipeline trainer (the version used by the RL loop):
- `selfplay.py:614` — `train_on_recent_games()` (full per-iteration trainer).
- `selfplay.py:690` — `_load_games()` (per-position weighting, draw-z, aux value blend) at lines 720–765.
- `selfplay.py:812` — column-flip augmentation block.
- `selfplay.py:849` — `_lr_lambda()` (warmup + cosine LR schedule).
- `selfplay.py:864` — main training loop (loss assembly at lines 868–890).
- `selfplay.py:917` — best-val state capture; restore at line 928.

Stand-alone supervised trainer (offline runs from the DB):
- `train.py:58` — `build_dataset()` (DB → tensors).
- `train.py:139` — `train()` (epoch loop, val tracking, best-checkpoint save).
- `train.py:206` — inner training step (loss = soft CE + MSE).

---

## 4. Reinforcement Learning Loop (self-play → train → gate)

The outer RL loop is the AlphaZero recipe: generate self-play games with
the current best net, train a candidate on a sliding window of recent
games, then accept the candidate only if it scores above a gating
threshold versus the previous best.

### Pseudocode

```
function run_pipeline(args):
    best_net = load(best.pt) or build_net()

    for it in 1..args.iterations:
        # --- 1. Self-play ---
        games = []
        repeat args.games_per_iter times in parallel workers:
            game = play_game(best_net, mcts_config)        # MCTS on every move
            games.append(game)
        save games to DB with model_version = "selfplay-v{it}"

        # --- 2. Training ---
        candidate = deep_copy(best_net)
        candidate, metrics = train_on_recent_games(
            candidate, db,
            window=args.window,
            epochs=args.epochs,
            extra_examples = pending_hard_examples,        # from last revert
        )

        # (Optional) PBT sibling: every K iterations, also train a
        # candidate with mutated lr/wd and pick the one with lower val loss.

        # --- 3. Gating / evaluation ---
        score = evaluate_nets(candidate, best_net,
                              num_games=args.eval_games,
                              add_noise=False, temperature=0)
        elo.update(version, best_version, score)
        if score > args.gate_threshold:
            best_net = candidate
            save promoted_{version}.pt                       # immutable snapshot
        save iter_{it}.pt and best.pt

        # --- 4. Periodic calibration tournament ---
        if it % args.tournament_every == 0:
            run_calibration_tournament(promoted_history + anchors)
            if champion_elo - current_best_elo > REVERT_GAP:
                best_net = champion
                pending_hard_examples = mine_hard_examples(champion, rejected)
```

The single self-play game:

```
function play_game(net, config):
    board = Board.initial()
    play `opening_random` random moves                       # data diversity
    next_root = None
    while not done and move_num < max_moves:
        root = MCTS(board, net, config,
                    add_noise=true, reuse_root=next_root)    # subtree reuse
        temp = 1.0 if move_num < temp_threshold else 0.0
        π    = get_policy(root, temp)                        # MCTS policy target
        a    = sample(π)
        record (board, π, a)
        next_root = root.children[a]                         # carry subtree
        board = board.apply(action_to_move(a))
        move_num += 1
    if no winner and adjudicate_gap > 0:
        winner = adjudicate by shortest-path differential
    return states, policies, actions, winner, final_board
```

Net-vs-alphabeta mixing — a fraction of games pit MCTS against the
alpha-beta engine to break self-imitation drift. Both NN moves and
alphabeta moves are recorded with the NN's MCTS visits as targets.

### Source references

Outer pipeline:
- `selfplay.py:1423` — `run_pipeline()` (top-level loop).
- `selfplay.py:1543` — main `for it` iteration body; phase headers at 1551, 1586, 1646.
- `selfplay.py:1693` — gating decision (`score > gate_threshold`).
- `selfplay.py:1701` — promoted-checkpoint persistence (`promoted_{version}.pt`).
- `selfplay.py:1614` — PBT sibling block.

Self-play game generation:
- `selfplay.py:245` — `play_game()` (MCTS-only self-play).
- `selfplay.py:176` — `play_game_vs_alphabeta()` (NN-vs-AB mix).
- `selfplay.py:286` — subtree-reuse plumbing (`next_root`).
- `selfplay.py:294` — temperature schedule (`temp_threshold`).
- `selfplay.py:89` — `_randomise_opening()` (opening diversity).
- `selfplay.py:110` — `adjudicate_winner()` (path-gap tiebreak).
- `selfplay.py:133` — `_draw_z()` (draw-value with stall + progress terms).
- `selfplay.py:350` — `generate_games()` (serial driver).
- `selfplay.py:515` — `generate_games_parallel()` (multiprocessing pool).
- `selfplay.py:459` — `_worker_play_one()` (per-worker game body).

Gating evaluation:
- `selfplay.py:940` — `evaluate_nets()` (serial; no Dirichlet noise, greedy after `eval_temp_moves`).
- `selfplay.py:1135` — `evaluate_nets_parallel()` (multiprocessing version).
- `selfplay.py:1211` — `wilson_ci()` (95% confidence interval on score).
- `selfplay.py:1232` — `EloTracker` (per-iteration K=32 Elo updates, JSON-persisted).

---

## 5. Round-Robin Tournament + Bradley–Terry Elo

Periodically, a round-robin between the recently promoted nets plus
historical anchors produces globally consistent Elos via Bradley–Terry
MLE. If the post-hoc champion is meaningfully ahead of the current best
(>25 Elo), the pipeline reverts to it.

### Pseudocode

```
function tournament(checkpoints, games_per_pair, sims):
    deduplicate by weight fingerprint (md5 of stem conv)
    jobs = []
    for each unordered pair (A, B):
        for k in 1..games_per_pair:
            colours alternate by k parity
            jobs.append((first_mover, second_mover, random_seed))

    wld = empty W/L/D table (one entry per unordered pair)
    parallel for each job in workers:
        winner, n_plies = play_greedy_mcts_game(net_a, net_b, sims, ...)
        wld[ canonical(a, b) ] += outcome

    matches = [(a, b, wins_a, wins_b, draws), ...]
    elos    = bradley_terry_mle(matches, anchor_label, anchor_rating=1000)
    elos_ci = bootstrap_resample(matches, n=200)
    save JSON with ratings + 95% CIs
    optionally save winning-side games into DB as "tourney-{champion}" rows
```

```
function bradley_terry_mle(matches, anchor, R0=1000, iters=3000, lr=1.0):
    R[p] = R0 for every player p
    scale = ln(10) / 400
    for _ in 1..iters:
        grad = {p: 0 for p in players}
        for (a, b, wa, wb, d) in matches:
            T   = wa + wb + d
            s_a = (wa + 0.5 * d) / T                     # observed score for a
            e_a = 1 / (1 + 10 ** ((R[b] - R[a]) / 400))  # expected score for a
            grad[a] += (s_a - e_a) * T
            grad[b] -= (s_a - e_a) * T
        for p in players:
            R[p] += lr * grad[p] / scale / N_players
        # Anchor: shift so R[anchor] stays at R0
        Δ = R0 - R[anchor]
        R[p] += Δ for every p
    return R
```

```
function bootstrap_elos(matches, n_resamples=200):
    point = bradley_terry_mle(matches)
    samples = {p: [] for p in players}
    repeat n_resamples times:
        for each pair (a, b) with T games:
            counts = multinomial(T, observed_proportions(W_a, W_b, D))
            resampled.append((a, b, *counts))
        boot = bradley_terry_mle(resampled)
        for p, r in boot: samples[p].append(r)
    return { p: (point[p], 2.5%-percentile, 97.5%-percentile) }
```

```
function play_greedy_mcts_game(net_a, net_b, sims):
    board = Board.initial()
    play opening_random uniform-random plies
    while not done and move < max_moves:
        net  = net_a if board.turn == 0 else net_b
        root = MCTS(board, net, MCTSConfig(num_simulations=sims,
                                           dirichlet_epsilon=0))
        action = argmax_a root.children[a].N             # greedy / temp 0
        board  = board.apply(action_to_move(action))
    if no winner: adjudicate by shortest-path gap
    return winner, move_count
```

The pipeline integrates the tournament by shelling out a subprocess and
parsing the JSON output, then comparing ratings:

```
if it % tournament_every == 0 and pool_size >= 2:
    ratings, champion = run_calibration_tournament(pool, anchor)
    if champion != current_best and ratings[champion] - ratings[best] > 25:
        rejected   = every promoted version since the champion
        hard_exs   = mine_hard_examples(champion_net, db, rejected)
                     # for each move in rejected games, run MCTS with the
                     # champion; if its top action differs, save that
                     # (state, soft_policy) pair as supervision
        best_net   = champion
        pending_hard_examples = hard_exs   # consumed in next training pass
```

### Source references

- `tournament.py:55` — `_play_game()` (one greedy-MCTS game between two nets).
- `tournament.py:120` — `_worker_play()` (pool worker: one job = one game).
- `tournament.py:106` — `_worker_init()` (loads each ckpt once per worker).
- `tournament.py:144` — `compute_elos()` (Bradley–Terry MLE by gradient descent on log-likelihood).
- `tournament.py:186` — `bootstrap_elos()` (multinomial resampling for 95% CIs).
- `tournament.py:268` — weight-fingerprint deduplication of identical checkpoints.
- `tournament.py:300` — pair schedule with alternating colours.
- `tournament.py:339` — outcome aggregation into the W/L/D matrix.
- `tournament.py:436` — optional `--save-to-db` of winner-only games as
  `tourney-{champion}` model_version rows (used as high-quality
  supervision in the next training pass).

Pipeline integration:
- `selfplay.py:1302` — `_run_calibration_tournament()` (subprocess shellout, JSON load).
- `selfplay.py:1786` — periodic-tournament trigger inside the RL loop.
- `selfplay.py:1836` — `REVERT_GAP_ELO = 25.0`; revert decision at lines 1840–1882.
- `selfplay.py:1346` — `_mine_hard_examples()` (positions where rejected vs champion diverge).

---

## 6. Deep Distillation

Two distillation tools transfer "deeper" knowledge into a smaller or
fresher network. Both treat distillation as supervised training on
`(state, π_teacher, v_teacher)` triples with KL-style policy loss + MSE
value loss, plus the catastrophic-forgetting mitigations recommended by
PROCESS.md §35.

### 6a. Shallow distill — net teacher (`distill.py`)

Used to seed a new architecture from an existing strong net (e.g. v74
6×64 → fresh 10×128 best.pt).

```
function shallow_distill(teacher, student, db, n_positions):
    boards = sample_positions(db, n_positions)
    states = stack(encode_state(b) for b in boards)
    π_T, v_T = teacher_targets(teacher, states)        # softmax(p_logits), v
    distill(teacher, student, states, π_T, v_T,
            epochs, batch_size, lr, weight_decay)

function distill(teacher, student, S, π_T, v_T, ...):
    split S/π_T/v_T into train + val
    optimizer = Adam(student.parameters, lr, weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * steps)
    for epoch in 1..epochs:
        for (x, π, v) in train_loader:
            p_logits, v_pred = student(x)
            log_p = log_softmax(p_logits)
            L     = -mean(sum(π * log_p, axis=1))      + MSE(v_pred, v)
            optimizer.step(); scheduler.step()
        track best_val, restore best weights at end
```

### 6b. Deep distill — search teacher + rehearsal + KL reg (`distill_deep.py`)

Replaces the teacher with one of:
- **mcts**: extreme-sim MCTS using the *student itself* as the value/policy
  net (e.g. 4–5k sims) — same net, deeper search.
- **ab**: alpha-beta search at high depth/time — fundamentally different
  evaluator; one-hot policy targets.

Adds the two PROCESS.md §35 mitigations against catastrophic forgetting:

```
function deep_distill(teacher_kind, student, db, n_positions, ...):
    boards = sample_positions(db, n_positions)

    # --- Teacher targets (mcts or ab) ---
    if teacher_kind == "mcts":
        examples = parallel_map(_mcts_teacher_one,
            (board, teacher_sims) for board in boards)
        # _mcts_teacher_one runs MCTS at high sims; returns
        # (state, get_policy(root, T=1.0), root.value)
    else:  # ab
        examples = parallel_map(_ab_teacher_one,
            (board, ab_depth, ab_time) for board in boards)
        # _ab_teacher_one returns (state, one_hot(best_move), 0.0)

    # --- Mitigation 2: rehearsal ---
    if rehearsal_frac > 0:
        n_rehearsal = n_teacher * rehearsal_frac / (1 - rehearsal_frac)
        examples += sample_rehearsal(db, n_rehearsal)
            # (state, π_from_stored_blob, ±1 from game outcome)

    # --- Mitigation 1: KL(reference || student) ---
    reference = frozen_copy(student)        # pre-distillation snapshot

    distill_student(student, examples, reference, reg_lambda):
        for (x, π, v) in batches:
            p_logits, v_pred = student(x)
            L = soft_CE(p_logits, π) + MSE(v_pred, v)
            if reg_lambda > 0:
                ref_logits, _ = reference(x)
                ref_p, ref_log_p = softmax / log_softmax(ref_logits)
                L += reg_lambda * mean(sum(ref_p * (ref_log_p - log_p)))
            backprop, step, scheduler
```

### 6c. Widening distill — wider student + dual-source teacher (`widen_distill.py`)

For when iterative training and depth-N AB distillation both saturate
because of capacity. Builds a wider net (e.g. 14×192 or 20×256) from
scratch and distils both:
1. The current 10×128 best (its policy/value on sampled positions).
2. Optionally fresh depth-8 alphabeta targets on the same positions.

```
function widen_distill(student_blocks, student_filters, teacher_net, ...):
    student = build_net(student_blocks, student_filters)   # blank wider net
    boards  = sample_positions(db, n_positions)

    net_examples = parallel_map(_net_teacher_one,
        (b, teacher_sims) for b in boards)
        # forward through teacher_net (MCTS); soft policy + scalar value

    if use_ab:
        ab_boards   = random.sample(boards, n_positions * ab_mix_frac)
        ab_examples = parallel_map(_ab_teacher_one,
            (b, ab_depth, ab_time) for b in ab_boards)
        examples = net_examples + ab_examples            # student sees both

    distill_into_wider(student, examples, ...,
                       reference=None, reg_lambda=0.0)   # blank student → no anchor
    save_checkpoint(student, ...)
```

### Source references

`distill.py` (shallow distill, single net teacher):
- `distill.py:36` — `sample_positions()` (replay random self-play games until N positions collected).
- `distill.py:63` — `teacher_targets()` (softmax(policy), value over batched forwards).
- `distill.py:77` — `distill()` (train loop with CosineAnnealingLR, best-val restore).
- `distill.py:116` — soft-CE + MSE loss assembly.

`distill_deep.py` (search teacher + rehearsal + KL reg):
- `distill_deep.py:59` — `sample_positions()`.
- `distill_deep.py:82` — `_mcts_teacher_one()` (high-sim MCTS teacher; uses `root.value` as v target).
- `distill_deep.py:98` — `_ab_teacher_one()` (alpha-beta teacher; one-hot policy, 0 value).
- `distill_deep.py:116` — `_mcts_init()` (per-worker net loader).
- `distill_deep.py:129` — `distill_student()` (loss = CE + MSE + reg_lambda·KL(ref||student)).
- `distill_deep.py:184` — KL reference-regularisation block (catastrophic-forgetting mitigation 1).
- `distill_deep.py:232` — `sample_rehearsal()` (mitigation 2: mix in self-play triples).
- `distill_deep.py:336` — rehearsal mixing in `main()` with `rehearsal_frac`.

`widen_distill.py` (wider student + dual-source teacher):
- `widen_distill.py:54` — `sample_positions()`.
- `widen_distill.py:81` — `_net_teacher_init()` / `_net_teacher_one()` (existing-net teacher via MCTS).
- `widen_distill.py:104` — `_ab_teacher_one()` (depth-8 AB co-teacher).
- `widen_distill.py:119` — `distill_into_wider()` (training loop; same loss structure as `distill_deep.distill_student`).
- `widen_distill.py:259` — `build_net(student_blocks, student_filters)` instantiation.
- `widen_distill.py:284` — AB co-teacher mix-in branch.

PROCESS.md §35 mitigations referenced above:
- KL regularisation toward a frozen pre-distillation copy of the student.
- Rehearsal: mix existing self-play (state, π, z) triples into the
  distillation training set so the student keeps its prior task visible.

---

## How the pieces connect

```
                ┌────────────────────────────────────────────┐
                │  RL loop (selfplay.py:run_pipeline)        │
                │                                            │
                │  ┌─────────┐   ┌──────────┐   ┌─────────┐  │
                │  │selfplay │ → │  train   │ → │  gate   │  │
                │  │ (MCTS)  │   │ on recent│   │ (eval)  │  │
                │  └─────────┘   └──────────┘   └─────────┘  │
                │       ▲              ▲             │       │
                │       │              │             ▼       │
                │   best_net      DB window     promote? ────┤
                │       │                             │      │
                │       │                             ▼      │
                │       │   every K iters:    tournament.py  │
                │       │   round-robin → BT MLE → revert?   │
                │       │                             │      │
                │       └─────────────────────────────┘      │
                └────────────────────────────────────────────┘

   Out-of-band tools (run manually when training saturates):
       distill.py        — net→net distill (architecture migration)
       distill_deep.py   — high-sim MCTS or deep AB → student
       widen_distill.py  — wider student inheriting net (+optional AB) targets

   Underneath:
       quoridor/mcts.py  — PUCT search shared by selfplay, eval, tournament,
                           hard-example mining, deep distillation
       quoridor/ai.py    — alpha-beta engine used as eval opponent and as a
                           distillation teacher
       quoridor/net.py   — residual policy/value network and (de)serialisation
```
