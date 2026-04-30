#!/usr/bin/env bash
# Launch the Tkinter GUI to play against the current best NN.
# gui.py loads checkpoints/best.pt by default and the start menu
# defaults to the "Neural Net" difficulty.

set -eu
cd "$(dirname "$0")"

exec python3 gui.py
