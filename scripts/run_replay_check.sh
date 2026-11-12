#!/usr/bin/env bash
# Replay check: r4vt action blend and r1rel pose blend at H=6, episodes 0 and 9.
# Usage: scripts/run_replay_check.sh --out DIR
set -euo pipefail

usage() {
  echo "usage: scripts/run_replay_check.sh --out DIR"
  echo "  replay_gt.py --mode runner --episodes 0 9 --horizon 6 --out DIR"
  echo "  refuses to overwrite an existing DIR"
}

die() {
  echo "error: $*" >&2
  exit 1
}

if [[ $# -ne 2 || "$1" != "--out" || -z "${2:-}" ]]; then
  usage
  exit 2
fi

source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-surface_frames \
  || die "could not source activate.sh"
# shellcheck disable=SC1091
source "$HOME/miniforge3/etc/profile.d/conda.sh" \
  || die "could not source conda.sh"
set +u
if ! conda activate pyrite; then
  set -u
  die "conda activate pyrite failed"
fi
set -u
export PYTHONPATH="$UMIFT_ROOT/UMIFT_Data/surface_frames_scripts:${PYTHONPATH}"

python "$UMIFT_ROOT/UMIFT_Data/surface_frames_scripts/replay_gt.py" \
  --mode runner --episodes 0 9 --horizon 6 --out "$2" \
  || die "replay_gt.py failed"
