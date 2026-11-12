#!/usr/bin/env bash
# Start the layup runner for one arm. The checkpoint is passed in UMIFT_CKPT_PATH.
# Usage: scripts/run_rollout.sh {r4vt|r1rel} [--replay-episode N]
set -euo pipefail

usage() {
  echo "usage: scripts/run_rollout.sh {r4vt|r1rel} [--replay-episode N]"
  echo "  r4vt   UMIFT_CKPT_PATH=/oct_v35_h4s_odom_r4vt/checkpoints/epoch=0185-train_loss=0.011.ckpt"
  echo "  r1rel  UMIFT_CKPT_PATH=/oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt"
  echo "  --replay-episode N   play that episode's labels through the runner (no policy call)"
  echo "  r4vt, policy and replay: jump guard holds a commanded step over 248.70 mm or 75.96 deg"
  echo "    and aborts after 3 holds in a row. r1rel is unchanged."
  echo "  r4vt, before the episode: moveL at <= 5 cm/s to 5 cm above the start along the mold"
  echo "    normal, Enter, descend to the start, Enter. Replay uses the recorded pose."
  echo "    Policy uses the current TCP. A failed check falls back to jogging."
}

die() {
  echo "error: $*" >&2
  exit 1
}

arm="${1:-}"
shift || true
replay_episode=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --replay-episode)
      [[ $# -ge 2 && -n "${2:-}" ]] || die "--replay-episode needs an episode id"
      replay_episode="$2"
      shift 2
      ;;
    *)
      usage
      die "unknown argument: $1"
      ;;
  esac
done

case "$arm" in
  r4vt)
    export UMIFT_CKPT_PATH="/oct_v35_h4s_odom_r4vt/checkpoints/epoch=0185-train_loss=0.011.ckpt"
    ;;
  r1rel)
    export UMIFT_CKPT_PATH="/oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt"
    ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ -n "$replay_episode" ]]; then
  export UMIFT_REPLAY_EPISODE="$replay_episode"
else
  unset UMIFT_REPLAY_EPISODE
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
export PYRITE_CHECKPOINT_FOLDERS=/mnt/data/umidata/ft/training_outputs
export PYRITE_HARDWARE_CONFIG_FOLDERS=/home/boeing/umi_ft_ws/hardware_interfaces/workcell/table_top_manip/config
export PYRITE_CONTROL_LOG_FOLDERS="${HOME}/data/control_log"
export PYTHONPATH="$UMIFT_ROOT/UMIFT_Data/surface_frames_scripts:${PYTHONPATH}"

python "$UMIFT_ROOT/PyriteEnvSuites/scripts/kill_ports.py" \
  || die "kill_ports.py failed"
python "$UMIFT_ROOT/PyriteEnvSuites/env_runners/umift_env_runner.py" \
  || die "umift_env_runner.py failed"
