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
  echo "  --foot-hyst          keep the foot face until a new face wins for N steps and is d mm closer."
  echo "                       N is UMIFT_FOOT_HYST_N (default 1), d is UMIFT_FOOT_HYST_D_MM (default 5)."
  echo "                       Off unless this flag or UMIFT_FOOT_FACE_HYST=1 is set."
  echo "  --spring25           load offline_hw/hwconfig_spring25 (max spring 25 N). Off keeps the 40 N yaml."
  echo "  --limited-vt         schedule the step-limited virtual target. Off rebuilds it from the foot."
  echo "  --max-step-mm M      cap each scheduled step at M mm. Default 25. The arm reaches ~30 mm/s."
  echo "  --mold-local         liftos only. robot0_eef_pos uses field_v2_axes. Off keeps fixture xyz."
  echo "  --no-mold-geom-train-frame"
  echo "                       publish the live smooth-field mold_geom. Default uses the"
  echo "                       training frame (frames_pca_r020.npz, PCA t1, 30 deg carry)."
  echo "  --newest-chunk       TCP-relative ΔR executes the newest chunk only. The 8 deg slew stays."
  echo "                       Off keeps the action-space blend. Ignored when rotation is not TCP-relative."
  echo "  --pose-ensemble      Integrate the whole chunk, then average overlapping SE3 targets."
  echo "                       Positions are averaged. Rotations use the quaternion mean. Decay 0.005."
  echo "                       The jump guard then checks the executed steps and the next blended target."
  echo "                       Off keeps the action-space blend and checks the whole decoded chunk."
  echo "                       Cannot be combined with --newest-chunk."
  echo "  --tcp-offset-mm N    Shift the tool frame N mm along tool z before foot and delta_n."
  echo "                       Commands sent to the arm are shifted back. Off leaves the pendant TCP."
  echo "  --dn-bias-mm N       Add N mm to scheduled δn when |F·n| ≥ 2 N or signed distance < 3 mm."
  echo "                       Off leaves δn unchanged. In the air the bias stays off."
  echo "  --exec-horizon N     execute N action steps, then replan. Off keeps 6."
  echo "  --slow-down N        wall time per action step is N/15 s. Off keeps 2 (0.133 s)."
  echo "                       N=6 loads offline_hw/hwconfig_spring25_sd6 (24 s history, 20% margin)."
  echo "  --hwconfig-dir PATH  hardware yaml folder. Off keeps spring25 or the shared yaml."
  echo "  --raw-replay-episode N  schedule that episode's recorded TCP. No policy and no mold foot."
  echo "  --raw-press          with --raw-replay-episode, press at -min(|F|, 25 N)/3000 when |F| > 5 N."
  echo "                       Off, nominal and virtual target are the same pose."
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
foot_hyst=0
spring25=0
limited_vt=0
max_step_mm=""
mold_local=0
no_train_frame=0
newest_chunk=0
pose_ensemble=0
tcp_offset_mm=""
dn_bias_mm=""
exec_horizon=""
slow_down=""
hwconfig_dir=""
raw_episode=""
raw_press=0
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
    --foot-hyst)
      foot_hyst=1
      shift
      ;;
    --spring25)
      spring25=1
      shift
      ;;
    --limited-vt)
      limited_vt=1
      shift
      ;;
    --max-step-mm)
      [[ $# -ge 2 && -n "${2:-}" ]] || die "--max-step-mm needs a length in mm"
      max_step_mm="$2"
      shift 2
      ;;
    --mold-local)
      mold_local=1
      shift
      ;;
    --no-mold-geom-train-frame)
      no_train_frame=1
      shift
      ;;
    --newest-chunk)
      newest_chunk=1
      shift
      ;;
    --pose-ensemble)
      pose_ensemble=1
      shift
      ;;
    --tcp-offset-mm)
      [[ $# -ge 2 && "${2:-}" =~ ^-?[0-9]+([.][0-9]+)?$ ]] || die "--tcp-offset-mm needs a number"
      tcp_offset_mm="$2"
      shift 2
      ;;
    --dn-bias-mm)
      [[ $# -ge 2 && "${2:-}" =~ ^-?[0-9]+([.][0-9]+)?$ ]] || die "--dn-bias-mm needs a number"
      dn_bias_mm="$2"
      shift 2
      ;;
    --exec-horizon)
      [[ $# -ge 2 && "${2:-}" =~ ^[0-9]+$ && "$2" -ge 1 ]] || die "--exec-horizon needs an integer >= 1"
      exec_horizon="$2"
      shift 2
      ;;
    --slow-down)
      [[ $# -ge 2 && "${2:-}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--slow-down needs a number > 0"
      slow_down="$2"
      awk "BEGIN { exit !(${slow_down} > 0) }" || die "--slow-down needs a number > 0"
      shift 2
      ;;
    --hwconfig-dir)
      [[ $# -ge 2 && -n "${2:-}" ]] || die "--hwconfig-dir needs a directory"
      hwconfig_dir="$2"
      shift 2
      ;;
    --raw-replay-episode)
      [[ $# -ge 2 && -n "${2:-}" ]] || die "--raw-replay-episode needs an episode id"
      raw_episode="$2"
      shift 2
      ;;
    --raw-press)
      raw_press=1
      shift
      ;;
    *)
      usage
      die "unknown argument: $1"
      ;;
  esac
done

if [[ "$pose_ensemble" -eq 1 && "$newest_chunk" -eq 1 ]]; then
  die "--pose-ensemble and --newest-chunk cannot be combined"
fi
if [[ "$pose_ensemble" -eq 1 && "$arm" != "r4vt" ]]; then
  die "--pose-ensemble is only for r4vt"
fi

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

if [[ -n "$replay_episode" && -n "$raw_episode" ]]; then
  die "--raw-replay-episode cannot be combined with --replay-episode"
fi
if [[ -n "$raw_press" && "$raw_press" -eq 1 && -z "$raw_episode" ]]; then
  die "--raw-press needs --raw-replay-episode"
fi
if [[ -n "$replay_episode" && -n "$start_episode" ]]; then
  die "--start-episode is for a policy run. Replay uses the recorded pose."
fi
if [[ -n "$start_index" && -z "$start_episode" && -z "$raw_episode" ]]; then
  die "--start-index needs --start-episode or --raw-replay-episode"
fi
if [[ -n "$start_episode" ]]; then
  export UMIFT_POLICY_START_EPISODE="$start_episode"
elif [[ -n "$raw_episode" ]]; then
  export UMIFT_POLICY_START_EPISODE="$raw_episode"
else
  unset UMIFT_POLICY_START_EPISODE
fi
if [[ -n "$start_index" ]]; then
  export UMIFT_POLICY_START_INDEX="$start_index"
else
  unset UMIFT_POLICY_START_INDEX
fi
if [[ -n "$raw_episode" ]]; then
  export UMIFT_RAW_POSE_REPLAY="$raw_episode"
  export UMIFT_RAW_POSE_START_INDEX="${start_index:-0}"
  if [[ "$raw_press" -eq 1 ]]; then
    export UMIFT_RAW_POSE_PRESS=1
  else
    unset UMIFT_RAW_POSE_PRESS
  fi
else
  unset UMIFT_RAW_POSE_REPLAY
  unset UMIFT_RAW_POSE_PRESS
  unset UMIFT_RAW_POSE_START_INDEX
fi
if [[ "$foot_hyst" -eq 1 ]]; then
  export UMIFT_FOOT_FACE_HYST=1
  export UMIFT_FOOT_HYST_N="${UMIFT_FOOT_HYST_N:-1}"
  export UMIFT_FOOT_HYST_D_MM="${UMIFT_FOOT_HYST_D_MM:-5}"
fi
if [[ "$limited_vt" -eq 1 ]]; then
  export UMIFT_SCHEDULE_LIMITED_VT=1
fi
if [[ -n "$max_step_mm" ]]; then
  export UMIFT_MAX_STEP_MM="$max_step_mm"
fi
if [[ "$no_train_frame" -eq 1 ]]; then
  export UMIFT_MOLD_GEOM_TRAIN_FRAME=0
else
  export UMIFT_MOLD_GEOM_TRAIN_FRAME="${UMIFT_MOLD_GEOM_TRAIN_FRAME:-1}"
fi
if [[ "$mold_local" -eq 1 ]]; then
  export UMIFT_MOLD_LOCAL_POS=1
  export UMIFT_MOLD_LOCAL_ZARR="${UMIFT_MOLD_LOCAL_ZARR:-/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_field_v2n_v3r_liftos/acp_replay_buffer.zarr}"
else
  unset UMIFT_MOLD_LOCAL_POS
fi
if [[ "$newest_chunk" -eq 1 ]]; then
  export UMIFT_TCP_ROT_NEWEST=1
else
  unset UMIFT_TCP_ROT_NEWEST
fi
if [[ "$pose_ensemble" -eq 1 ]]; then
  export UMIFT_POSE_ENSEMBLE=1
else
  unset UMIFT_POSE_ENSEMBLE
fi
if [[ -n "$tcp_offset_mm" ]]; then
  export UMIFT_TCP_OFFSET_MM="$tcp_offset_mm"
else
  unset UMIFT_TCP_OFFSET_MM
fi
if [[ -n "$dn_bias_mm" ]]; then
  export UMIFT_DN_BIAS_MM="$dn_bias_mm"
else
  unset UMIFT_DN_BIAS_MM
fi
if [[ -n "$exec_horizon" ]]; then
  export UMIFT_EXEC_HORIZON="$exec_horizon"
else
  unset UMIFT_EXEC_HORIZON
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
_SD6_HW="/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/offline_hw/hwconfig_spring25_sd6"
if [[ -n "$hwconfig_dir" ]]; then
  [[ -d "$hwconfig_dir" ]] || die "hwconfig dir not found: $hwconfig_dir"
  export PYRITE_HARDWARE_CONFIG_FOLDERS="$hwconfig_dir"
elif [[ -n "$slow_down" ]] && awk "BEGIN { exit !(${slow_down} == 6) }"; then
  export PYRITE_HARDWARE_CONFIG_FOLDERS="$_SD6_HW"
elif [[ "$spring25" -eq 1 ]]; then
  export PYRITE_HARDWARE_CONFIG_FOLDERS="/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/offline_hw/hwconfig_spring25"
fi
echo "hardware config: $PYRITE_HARDWARE_CONFIG_FOLDERS"
if [[ -n "$slow_down" ]]; then
  export UMIFT_SLOW_DOWN_FACTOR="$slow_down"
else
  export UMIFT_SLOW_DOWN_FACTOR="${UMIFT_SLOW_DOWN_FACTOR:-2}"
fi
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
