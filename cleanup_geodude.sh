#!/bin/bash
# Review, then run on geodude-control. Not executed by the audit.
# Vat candidates only, about 25G. ASK rows, other users, and standard_bc/ffs_depth/main are absent.
set -u
M=/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr
T=/mnt/data/umidata/ft/training_outputs

# True when a running Hyak job command or sbatch mentions this exact string.
job_reads() {
  local needle="$1"
  ssh -o BatchMode=yes vtevy@klone.hyak.uw.edu "bash -s" -- "$needle" << 'EOF'
needle="$1"
while read -r j; do
  [[ -n "$j" ]] || continue
  cmd=$(scontrol show job "$j" | sed -n 's/.*Command=\([^ ]*\).*/\1/p')
  [[ "$cmd" == *"$needle"* ]] && exit 0
  [[ -f "$cmd" ]] && grep -qF -- "$needle" "$cmd" && exit 0
done < <(squeue -u vtevy -h -o '%i')
exit 1
EOF
}

# 3.4G tar, 3580805120 bytes. Delete only if the extracted zarr is still there.
tar="$M/r4vt_causal_foot.tar"
if [[ -d "$M/r4vt_causal_foot/acp_replay_buffer.zarr" && $(stat -c %s "$tar") == 3580805120 ]]; then
  rm "$tar"
else echo "skip $tar"; fi

# 920M truncated heading e199. A good checkpoint is 3047598380 bytes.
f="$T/oct-v35-h4s-r4vt-heading-tilt_sc_e200_bs64_valtopk_43448962/checkpoints/epoch=0199-val_loss=0.237.ckpt"
if [[ -f "$f" && $(stat -c %s "$f") == 964689920 ]]; then
  rm "$f"
else echo "skip $f"; fi

# 2.9G causal epoch 120. Same 3047655148 bytes as the kept e25/e40/latest files.
f="$T/oct_v35_h4s_r4vt_causal_foot/oct-v35-h4s-r4vt-causal-foot_e200_bs64_valtopk_43390127/checkpoints/epoch=0120-val_loss=0.148.ckpt"
if [[ -f "$f" && $(stat -c %s "$f") == 3047655148 ]]; then
  rm "$f"
else echo "skip $f"; fi

# 78M pre-k3000 carry zarr. The live job reads r4vt_causal_foot_carry_k3000.
d="$M/r4vt_causal_foot_carry"
if [[ ! -d "$M/r4vt_causal_foot_carry_k3000/acp_replay_buffer.zarr" ]]; then echo "skip $d"
elif job_reads "r4vt_causal_foot_carry/"; then echo "skip $d"
else
  rm -r "$d"
fi

# 2.9G smooth-frame run. Finished, also on Hyak.
d="$T/oct_v35_h4s_r4vt_smooth_frame"
if job_reads 43335222; then echo "skip $d"; else rm -r "$d"; fi

# 2.9G frame-continuity run. Finished.
d="$T/oct_v35_h4s_r4vt_frame_continuity_causal"
if job_reads 43308311; then echo "skip $d"; else rm -r "$d"; fi

# 5.7G step-mask run. Finished.
d="$T/oct_v35_h4s_r4vt_track_clean_step_mask"
if job_reads 43363017; then echo "skip $d"; else rm -r "$d"; fi

# 6.6G train image v7, 7126274048 bytes, same size as v8.
f=/mnt/data/umidata/ft/images/umift-train_v7.sif
if [[ -f /mnt/data/umidata/ft/images/umift-train_v8.sif \
   && $(stat -c %s "$f") == 7126274048 \
   && $(stat -c %s /mnt/data/umidata/ft/images/umift-train_v8.sif) == 7126274048 ]]; then
  rm "$f"
else echo "skip $f"; fi
