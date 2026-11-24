#!/usr/bin/env bash
# Start the layup runner for one arm. The checkpoint is passed in UMIFT_CKPT_PATH.
# Usage: scripts/run_rollout.sh {r4vt|r1rel} [--replay-episode N] [--extrinsics PATH] [--ckpt PATH] [--bench] [--bench-lift M] [--out_dir PATH] [--start-episode N] [--start-index I] [--dry-run]
set -euo pipefail
_orig=("$@")

usage() {
  echo "usage: scripts/run_rollout.sh {r4vt|r1rel} [--replay-episode N] [--extrinsics PATH] [--ckpt PATH] [--bench] [--bench-lift M] [--out_dir PATH] [--name NAME] [--start-episode N] [--start-index I] [--dry-run]"
  echo "  r4vt   track_clean epoch 55, unless UMIFT_CKPT_PATH or --ckpt is set."
  echo "         UMIFT_MAX_DURATION_S overrides the 8 min cap."
  echo "  r1rel  UMIFT_CKPT_PATH=/oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt"
  echo "  --replay-episode N   play that episode's labels through the runner (no policy call)"
  echo "  --extrinsics PATH    set UMIFT_WORKCELL_EXTRINSICS to that yaml"
  echo "  --ckpt PATH          checkpoint path relative to PYRITE_CHECKPOINT_FOLDERS, leading /"
  echo "  --bench              first free-space replay: clamp delta_n at -5 mm"
  echo "  UMIFT_STIFFNESS_FLOOR=3000   press-axis floor at schedule time (off if unset)"
  echo "  UMIFT_BENCH_DN_FLOOR_MM=-5   bench δn floor in mm (default -5, needs --bench)"
  echo "  --bench-lift M       shift foot, virtual target, and start pose M metres"
  echo "                       along fixture up (−Y, away from the table)"
  echo "  r4vt replay reads r4vt_causal_foot (10-d) unless UMIFT_R4VT_ZARR is set."
  echo "  r4vt policy reads UMIFT_R4VT_ZARR (default: r4vt_smooth_tcp_track_clean) and the"
  echo "    2026-11-16 orange extrinsics. UMIFT_ABORT_ON_MEASURED_MESH_DISTANCE=1."
  echo "  r4vt, policy and replay: jump guard holds a commanded step over 248.70 mm or 75.96 deg"
  echo "    and aborts after 3 holds in a row. r1rel is unchanged."
  echo "  r4vt, before the episode: moveL at <= 5 cm/s to 5 cm back along the tool axis,"
  echo "    Enter, then straight down that axis. The episode log includes the move."
  echo "    Replay uses the recorded pose."
  echo "  --out_dir PATH       write this run's logs, video, meta.json, and notes.md there."
  echo "  --name NAME          with UMIFT_RESULTS_ROOT, use \$ROOT/rollouts/<today>_NAME."
  echo "    Without either, logs stay in \$PYRITE_CONTROL_LOG_FOLDERS (\$HOME/data/control_log)."
  echo "  --start-episode N    policy only. Approach episode N. Replay ignores this and"
  echo "    refuses the flag, because replay uses the recorded pose."
  echo "  --start-index I      policy only. Row of ts_pose_fb_0. Default 0 (first frame)."
  echo "  --dry-run            parse, write meta.json if --out_dir/--name is set, then exit."
  echo "    Does not call kill_ports.py or the runner."
  echo "    Policy uses the current TCP, unless UMIFT_POLICY_START_EPISODE=N."
  echo "    That approaches episode N's TCP pose (index UMIFT_POLICY_START_INDEX, default 0), and"
  echo "    holds 8 s before the first query. If that TCP is inside the mold,"
  echo "    the descent stops at the surface and the normal gap is allowed."
  echo "    Ignored for a GT replay."
  echo "    A failed check falls back to jogging, except the policy start, which aborts."
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
out_dir=""
run_name=""
start_episode=""
start_index=""
dry=0
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
    --out_dir)
      [[ $# -ge 2 && -n "${2:-}" ]] || die "--out_dir needs a directory"
      out_dir="$2"
      shift 2
      ;;
    --name)
      [[ $# -ge 2 && -n "${2:-}" ]] || die "--name needs a run name"
      run_name="$2"
      shift 2
      ;;
    --start-episode)
      [[ $# -ge 2 && -n "${2:-}" ]] || die "--start-episode needs an episode id"
      start_episode="$2"
      shift 2
      ;;
    --start-index)
      [[ $# -ge 2 && -n "${2:-}" ]] || die "--start-index needs a frame index"
      start_index="$2"
      shift 2
      ;;
    --dry-run)
      dry=1
      shift
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

if [[ -n "$replay_episode" && -n "$start_episode" ]]; then
  die "--start-episode is for a policy run. Replay uses the recorded pose."
fi
if [[ -n "$start_index" && -z "$start_episode" ]]; then
  die "--start-index needs --start-episode"
fi
if [[ -n "$start_episode" ]]; then
  export UMIFT_POLICY_START_EPISODE="$start_episode"
else
  unset UMIFT_POLICY_START_EPISODE
fi
if [[ -n "$start_index" ]]; then
  export UMIFT_POLICY_START_INDEX="$start_index"
else
  unset UMIFT_POLICY_START_INDEX
fi

# Policy labels stay track_clean. A replay reads the causal-foot 10-d labels.
_ASSETS="/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/UMIFT_Data/surface_frames_scripts/assets"
_CAUSAL_ZARR="/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_causal_foot/acp_replay_buffer.zarr"
_CAUSAL_NORM="$_ASSETS/oct_v35_h4s_r4vt_action_normalizer_minmax_causal_foot.pkl"
_CAUSAL_SHA="14308b72d55600b4359dfa67d10aedab378e42ad26f2bbcac988eed4d97a1ac2"
_TRACK_ZARR="/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_smooth_tcp_track_clean/acp_replay_buffer.zarr"
_TRACK_NORM="$_ASSETS/oct_v35_h4s_r4vt_action_normalizer_minmax_track_clean.pkl"
_TRACK_SHA="29a48f69ff51807df631b1e350087d6991091cba9739f0be88b30e2c85cf3bd4"
if [[ -n "$replay_episode" && "$arm" == "r4vt" ]]; then
  export UMIFT_R4VT_ZARR="${UMIFT_R4VT_ZARR:-$_CAUSAL_ZARR}"
  export UMIFT_R4VT_NORMALIZER="${UMIFT_R4VT_NORMALIZER:-$_CAUSAL_NORM}"
  export UMIFT_R4VT_NORMALIZER_SHA="${UMIFT_R4VT_NORMALIZER_SHA:-$_CAUSAL_SHA}"
  export UMIFT_ACTION_NORMALIZER="${UMIFT_ACTION_NORMALIZER:-$_CAUSAL_NORM}"
  export UMIFT_ACTION_NORMALIZER_SHA="${UMIFT_ACTION_NORMALIZER_SHA:-$_CAUSAL_SHA}"
else
  export UMIFT_R4VT_ZARR="${UMIFT_R4VT_ZARR:-$_TRACK_ZARR}"
  export UMIFT_ACTION_NORMALIZER="${UMIFT_ACTION_NORMALIZER:-$_TRACK_NORM}"
  export UMIFT_ACTION_NORMALIZER_SHA="${UMIFT_ACTION_NORMALIZER_SHA:-$_TRACK_SHA}"
fi
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

export UMIFT_ALIGN_OBS_TO_IMAGE="${UMIFT_ALIGN_OBS_TO_IMAGE:-1}"
export UMIFT_FOOT_WALK="${UMIFT_FOOT_WALK:-geodesic}"
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
export UMIFT_SLOW_DOWN_FACTOR="${UMIFT_SLOW_DOWN_FACTOR:-2}"
if [[ -n "$run_name" && -z "$out_dir" ]]; then
  [[ -n "${UMIFT_RESULTS_ROOT:-}" ]] || die "--name needs UMIFT_RESULTS_ROOT"
  out_dir="${UMIFT_RESULTS_ROOT}/rollouts/$(TZ=America/Los_Angeles date +%F)_${run_name}"
fi
if [[ -n "$out_dir" ]]; then
  mkdir -p "$out_dir"
  export UMIFT_RUN_DIR="$(realpath "$out_dir")"
  export PYRITE_CONTROL_LOG_FOLDERS="$UMIFT_RUN_DIR"
else
  unset UMIFT_RUN_DIR
  export PYRITE_CONTROL_LOG_FOLDERS="${HOME}/data/control_log"
fi
export PYTHONPATH="$UMIFT_ROOT/UMIFT_Data/surface_frames_scripts:${PYTHONPATH}"
export UMIFT_RUN_COMMAND="$(printf '%q ' "$0" "${_orig[@]}")"

if [[ -n "${UMIFT_RUN_DIR:-}" ]]; then
  python -m PyriteEnvSuites.env_runners.run_meta \
    || die "could not write meta.json"
fi
if [[ "$dry" -eq 1 ]]; then
  echo "dry-run: parsed. not starting the robot."
  exit 0
fi

python "$UMIFT_ROOT/PyriteEnvSuites/scripts/kill_ports.py" \
  || die "kill_ports.py failed"
python "$UMIFT_ROOT/PyriteEnvSuites/env_runners/umift_env_runner.py" \
  || die "umift_env_runner.py failed"
