"""Copy all human-vs-NN games from quoridor.db into quoridor_v3.db
so future training pulls from one source.  Idempotent: skips games
already in v3.
"""
import sqlite3
from quoridor import Board, GameDB
from quoridor.encoding import serialize_policy

src = sqlite3.connect("data/quoridor.db")
dst = GameDB("data/quoridor_v3.db")

# Source: gui-nn games where one side is human
src_games = src.execute(
    "SELECT id, winner, p1_source, p2_source, model_version, notes "
    "FROM games WHERE model_version LIKE 'gui-nn%' "
    "AND (p1_source='human' OR p2_source='human') "
    "ORDER BY id"
).fetchall()
print(f"Source has {len(src_games)} human-vs-NN games")

# Destination: check what's already there to avoid duplicates by num_plies + winner
dst_existing = list(
    dst.iter_games(finished_only=False)
)
dst_known_keys = set()
for r in dst_existing:
    if r[7] and r[7].startswith("gui-nn"):
        dst_known_keys.add((r[3], r[4]))  # (winner, num_plies)
print(f"Destination already has {len(dst_known_keys)} gui-nn games")

copied = 0
for src_gid, winner, p1s, p2s, mv, notes in src_games:
    # Load moves from source
    moves_cur = src.execute(
        "SELECT move_kind, move_r, move_c FROM moves WHERE game_id=? ORDER BY ply",
        (src_gid,)
    )
    from quoridor.board import Move
    moves = [Move(k, r, c) for (k, r, c) in moves_cur.fetchall()]
    if not moves:
        continue
    if (winner, len(moves)) in dst_known_keys:
        print(f"  skip {src_gid}: already in v3 (w={winner}, plies={len(moves)})")
        continue
    # Save
    new_gid = dst.save_game(
        moves,
        winner=winner,
        p1_source=p1s,
        p2_source=p2s,
        model_version=mv,
        notes=notes or "",
    )
    print(f"  copied src id={src_gid} -> v3 id={new_gid}  "
          f"(plies={len(moves)}, winner=P{winner+1 if winner is not None else '?'})")
    copied += 1

dst.close()
print(f"\nCopied {copied} new games. v3 now has all human-vs-NN games.")
