#!/usr/bin/env bash
# Start the layup runner for one arm. The checkpoint is passed in UMIFT_CKPT_PATH.
# Usage: scripts/run_rollout.sh {r4vt|r1rel} [--replay-episode N] [--extrinsics PATH] [--ckpt PATH] [--bench] [--bench-lift M]
set -euo pipefail

usage() {
  echo "usage: scripts/run_rollout.sh {r4vt|r1rel} [--replay-episode N] [--extrinsics PATH] [--ckpt PATH] [--bench] [--bench-lift M]"
  echo "  r4vt   track_clean epoch 55, unless UMIFT_CKPT_PATH or --ckpt is set."
  echo "         UMIFT_MAX_DURATION_S overrides the 8 min cap."
  echo "  r1rel  UMIFT_CKPT_PATH=/oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt"
  echo "  --replay-episode N   play that episode's labels through the runner (no policy call)"
  echo "  --extrinsics PATH    set UMIFT_WORKCELL_EXTRINSICS to that yaml"
  echo "  --ckpt PATH          checkpoint path relative to PYRITE_CHECKPOINT_FOLDERS, leading /"
  echo "  --bench              first free-space replay: clamp delta_n at -5 mm"
  echo "  --bench-lift M       shift foot, virtual target, and start pose M metres"
  echo "                       along fixture up (−Y, away from the table)"
  echo "  r4vt reads UMIFT_R4VT_ZARR (default: r4vt_smooth_tcp_track_clean) and the"
  echo "    2026-11-16 orange extrinsics. UMIFT_ABORT_ON_MEASURED_MESH_DISTANCE=1."
  echo "  r4vt, policy and replay: jump guard holds a commanded step over 248.70 mm or 75.96 deg"
  echo "    and aborts after 3 holds in a row. r1rel is unchanged."
  echo "  r4vt, before the episode: moveL at <= 5 cm/s to 5 cm back along the tool axis,"
  echo "    Enter, then straight down that axis. The episode log includes the move."
  echo "    Replay uses the recorded pose."
  echo "    Policy uses the current TCP. A failed check falls back to jogging."
}

die() {
  echo "error: $*" >&2
  exit 1
}

arm="${1:-}"
shift || true
replay_episode=""
extrinsics=""
ckpt=""
bench=0
bench_lift=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --replay-episode)
      [[ $# -ge 2 && -n "${2:-}" ]] || die "--replay-episode needs an episode id"
      replay_episode="$2"
      shift 2
      ;;
    --extrinsics)
      [[ $# -ge 2 && -n "${2:-}" ]] || die "--extrinsics needs a yaml path"
      extrinsics="$2"
      shift 2
      ;;
    --ckpt)
      [[ $# -ge 2 && -n "${2:-}" ]] || die "--ckpt needs a checkpoint path"
      ckpt="$2"
      shift 2
      ;;
    --bench)
      bench=1
      shift
      ;;
    --bench-lift)
      [[ $# -ge 2 && -n "${2:-}" ]] || die "--bench-lift needs a distance in metres"
      bench_lift="$2"
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
    export UMIFT_CKPT_PATH="${UMIFT_CKPT_PATH:-/oct_v35_h4s_r4vt_track_clean/oct-v35-h4s-r4vt-track-clean_e100_bs64_valtopk_43351238/checkpoints/epoch=0055-val_loss=0.030.ckpt}"
    ;;
  r1rel)
    export UMIFT_CKPT_PATH="${UMIFT_CKPT_PATH:-/oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt}"
    ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ -n "$ckpt" ]]; then
  export UMIFT_CKPT_PATH="$ckpt"
fi

if [[ -n "$extrinsics" ]]; then
  [[ -f "$extrinsics" ]] || die "extrinsics file not found: $extrinsics"
  export UMIFT_WORKCELL_EXTRINSICS="$(realpath "$extrinsics")"
fi

if [[ -n "$replay_episode" ]]; then
  export UMIFT_REPLAY_EPISODE="$replay_episode"
else
  unset UMIFT_REPLAY_EPISODE
fi

# Track-clean labels. The runner compares the checkpoint normalizer to this pickle.
export UMIFT_R4VT_ZARR="${UMIFT_R4VT_ZARR:-/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_smooth_tcp_track_clean/acp_replay_buffer.zarr}"
export UMIFT_ACTION_NORMALIZER="${UMIFT_ACTION_NORMALIZER:-/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax_track_clean.pkl}"
export UMIFT_ACTION_NORMALIZER_SHA="${UMIFT_ACTION_NORMALIZER_SHA:-29a48f69ff51807df631b1e350087d6991091cba9739f0be88b30e2c85cf3bd4}"
export UMIFT_WORKCELL_EXTRINSICS="${UMIFT_WORKCELL_EXTRINSICS:-/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/PyriteConfig/tasks/umift/layup_workcell_extrinsics_20261116_orange.yaml}"
export UMIFT_ABORT_ON_MEASURED_MESH_DISTANCE="${UMIFT_ABORT_ON_MEASURED_MESH_DISTANCE:-1}"
# --bench is the first free-space replay: clamp delta_n at -5 mm.
# Leave it off for a later contact replay.
if [[ "$bench" -eq 1 ]]; then
  export UMIFT_ENABLE_DELTA_N_FLOOR=1
fi
# Fixture +Y is down. This raises the replay off the table.
if [[ -n "$bench_lift" ]]; then
  export UMIFT_BENCH_LIFT_M="$bench_lift"
else
  unset UMIFT_BENCH_LIFT_M
fi
if [[ ! -f "$UMIFT_WORKCELL_EXTRINSICS" ]]; then
  die "orange extrinsics not found: $UMIFT_WORKCELL_EXTRINSICS"
fi
if [[ "$arm" == "r4vt" ]]; then
  if [[ ! -f "$UMIFT_ACTION_NORMALIZER" ]]; then
    die "track_clean normalizer not found: $UMIFT_ACTION_NORMALIZER"
  fi
  _sidecar="/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_smooth_frames.npz"
  if [[ ! -f "$_sidecar" ]]; then
    die "smooth sidecar not found: $_sidecar"
  fi
  echo "max_duration_s=${UMIFT_MAX_DURATION_S:-480}"
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
