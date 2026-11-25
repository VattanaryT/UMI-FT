#!/bin/bash
# Fail closed before an r4vt rollout. The liftos normalizer, the training
# orientation file, and train-frame mold_geom have to match the checkpoint.
set -euo pipefail

SHA_EXPECT="ce85fae23a90d3d7fc29d2bc357693b17254fbadea1af716ba2f10af9be394af"
ORI_PREFIX="fa220db41eca8ac3"
ORI="/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_field_v2_k3000/frames_pca_r020.npz"
SD6_HW="/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/offline_hw/hwconfig_spring25_sd6"

fail() {
  echo "preflight FAIL: $*" >&2
  exit 1
}

norm="${UMIFT_ACTION_NORMALIZER:-}"
[[ -n "$norm" && -f "$norm" ]] || fail "normalizer missing (${norm:-unset})"
got="$(sha256sum "$norm" | awk '{print $1}')"
[[ "$got" == "$SHA_EXPECT" ]] || fail "normalizer sha ${got} != ${SHA_EXPECT}"
echo "preflight OK: normalizer ${got}"

[[ -f "$ORI" ]] || fail "orientation file missing: ${ORI}"
osha="$(sha256sum "$ORI" | awk '{print $1}')"
[[ "$osha" == "${ORI_PREFIX}"* ]] || fail "orientation sha ${osha} does not start with ${ORI_PREFIX}"
echo "preflight OK: orientation ${osha}"

tf="${UMIFT_MOLD_GEOM_TRAIN_FRAME:-1}"
case "$tf" in
  0|false|no|off|False|FALSE|NO|OFF)
    fail "mold_geom train_frame is off (${tf})"
    ;;
esac
echo "preflight OK: mold_geom train_frame on"

off="${UMIFT_TCP_OFFSET_MM:-0}"
awk "BEGIN { exit !(${off} == 0) }" || fail "tcp offset is ${off}, want 0"
echo "preflight OK: tcp offset 0; pendant TCP is 156 mm"

sd="${UMIFT_SLOW_DOWN_FACTOR:-2}"
hw="${PYRITE_HARDWARE_CONFIG_FOLDERS:-}"
[[ -n "$hw" ]] || fail "PYRITE_HARDWARE_CONFIG_FOLDERS unset"
if awk "BEGIN { exit !(${sd} == 6) }"; then
  [[ "$hw" == "$SD6_HW" ]] || fail "slow-down 6 needs ${SD6_HW}, got ${hw}"
else
  [[ "$hw" != "$SD6_HW" ]] || fail "slow-down ${sd} must not use the sd6 hwconfig"
fi
[[ -f "${hw}/ur5e_wrist_ft.yaml" ]] || fail "missing ${hw}/ur5e_wrist_ft.yaml"
echo "preflight OK: hwconfig ${hw} slow-down ${sd}"

ckpt="${UMIFT_CKPT_PATH:-}"
[[ -n "$ckpt" ]] || fail "UMIFT_CKPT_PATH unset"
root="${PYRITE_CHECKPOINT_FOLDERS:-}"
found=""
if [[ -f "$ckpt" ]]; then
  found="$ckpt"
elif [[ -n "$root" && -f "${root}${ckpt}" ]]; then
  found="${root}${ckpt}"
elif [[ -n "$root" && -f "${root}/${ckpt#/}" ]]; then
  found="${root}/${ckpt#/}"
fi
[[ -n "$found" ]] || fail "ckpt not found: ${ckpt} (checkpoint folder ${root:-unset})"
echo "preflight OK: ckpt ${found}"
