#!/usr/bin/env bash
# Stage 2 queue (README): run the given "<condition> <seed>" pairs one after
# another, each with launch.py; a failed run is logged and the queue goes on.
# Two lanes side by side (14 simulation workers each), e.g. from the
# PredPreyGrass-archive root (PY: a Python with PredPreyGrass installed):
#   systemd-run --user --unit=sw-lane1 --working-directory=$PWD -E PY=<python> \
#       bash foraging_swarms/launchers/queue_stage2.sh "B 1" "C 2" "B 3" "A 1"
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"
cd "$HERE/../.."  # archive root, so foraging_swarms is importable
LOGS="$HERE/../runs/logs"
mkdir -p "$LOGS"
for entry in "$@"; do
    set -- $entry
    log="$LOGS/train_${1}_seed${2}_$(date +%Y-%m-%d_%H-%M-%S).log"
    echo "$(date '+%F %T') start $1 seed $2 -> $log" >> "$LOGS/queue.log"
    "$PY" -m foraging_swarms.launchers.launch "$1" "$2" > "$log" 2>&1
    echo "$(date '+%F %T') end $1 seed $2 exit $?" >> "$LOGS/queue.log"
done
