#!/usr/bin/env bash
# Start the layup runner for one arm. The checkpoint is passed in UMIFT_CKPT_PATH.
# Usage: scripts/run_rollout.sh {r4vt|r1rel} [--replay-episode N] [--bench] [--bench-lift M]
set -euo pipefail

usage() {
  echo "usage: scripts/run_rollout.sh {r4vt|r1rel} [--replay-episode N] [--bench] [--bench-lift M]"
  echo "  r4vt   UMIFT_CKPT_PATH=/oct_v35_h4s_odom_r4vt/checkpoints/epoch=0185-train_loss=0.011.ckpt"
  echo "  r1rel  UMIFT_CKPT_PATH=/oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt"
  echo "  --replay-episode N   play that episode's labels through the runner (no policy call)"
  echo "  --bench              first free-space replay: clamp delta_n at -5 mm"
  echo "  --bench-lift M       shift foot, virtual target, and start pose M metres"
  echo "                       along fixture up (−Y, away from the table)"
  echo "  r4vt reads UMIFT_R4VT_ZARR (default: r4vt_frame_continuity_causal) and the"
  echo "    2026-11-16 orange extrinsics. UMIFT_ABORT_ON_MEASURED_MESH_DISTANCE=1."
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
bench=0
bench_lift=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --replay-episode)
      [[ $# -ge 2 && -n "${2:-}" ]] || die "--replay-episode needs an episode id"
      replay_episode="$2"
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
    export UMIFT_CKPT_PATH="/oct_v35_h4s_r4vt_frame_continuity_causal/oct-v35-h4s-r4vt-causal_e100_bs64_valtopk_43308311/checkpoints/epoch=0030-val_loss=0.032.ckpt"
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

# Causal labels and this worktree's runner. surface_frames is left unchanged.
export UMIFT_R4VT_ZARR="${UMIFT_R4VT_ZARR:-/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_frame_continuity_causal/acp_replay_buffer.zarr}"
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

source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-frame_continuity \
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
