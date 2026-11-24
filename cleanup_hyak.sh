#!/bin/bash
# Review, then run on a Hyak login node as vtevy. Not executed by the audit.
# Scrubbed candidates ~117G. Weirdlab candidates ~235G. ASK rows are absent.
set -u
S=/gscratch/scrubbed/vtevy
W=/gscratch/weirdlab/vtevy

# A running job reads this name if its sbatch command or script text contains it.
job_reads() {
  local needle="$1" j cmd
  while read -r j; do
    [[ -n "$j" ]] || continue
    cmd=$(scontrol show job "$j" | sed -n 's/.*Command=\([^ ]*\).*/\1/p')
    [[ "$cmd" == *"$needle"* ]] && return 0
    [[ -f "$cmd" ]] && grep -qF -- "$needle" "$cmd" && return 0
  done < <(squeue -u vtevy -h -o '%i')
  return 1
}

# 2.84G each (3047655148 bytes). Causal-foot 43390127 epochs other than e25, e40, and latest.
C="$S/training_outputs/oct_v35_h4s_r4vt_causal_foot/oct-v35-h4s-r4vt-causal-foot_e200_bs64_valtopk_43390127/checkpoints"
f="$C/epoch=0000-val_loss=0.325.ckpt"
if [[ -f "$f" && $(stat -c %s "$f") == 3047655148 ]]; then rm "$f"; else echo "skip $f"; fi
f="$C/epoch=0005-val_loss=0.080.ckpt"
if [[ -f "$f" && $(stat -c %s "$f") == 3047655148 ]]; then rm "$f"; else echo "skip $f"; fi
f="$C/epoch=0010-val_loss=0.069.ckpt"
if [[ -f "$f" && $(stat -c %s "$f") == 3047655148 ]]; then rm "$f"; else echo "skip $f"; fi
f="$C/epoch=0015-val_loss=0.067.ckpt"
if [[ -f "$f" && $(stat -c %s "$f") == 3047655148 ]]; then rm "$f"; else echo "skip $f"; fi
f="$C/epoch=0020-val_loss=0.061.ckpt"
if [[ -f "$f" && $(stat -c %s "$f") == 3047655148 ]]; then rm "$f"; else echo "skip $f"; fi
f="$C/epoch=0030-val_loss=0.062.ckpt"
if [[ -f "$f" && $(stat -c %s "$f") == 3047655148 ]]; then rm "$f"; else echo "skip $f"; fi
f="$C/epoch=0035-val_loss=0.061.ckpt"
if [[ -f "$f" && $(stat -c %s "$f") == 3047655148 ]]; then rm "$f"; else echo "skip $f"; fi
f="$C/epoch=0060-val_loss=0.083.ckpt"
if [[ -f "$f" && $(stat -c %s "$f") == 3047655148 ]]; then rm "$f"; else echo "skip $f"; fi
f="$C/epoch=0080-val_loss=0.098.ckpt"
if [[ -f "$f" && $(stat -c %s "$f") == 3047655148 ]]; then rm "$f"; else echo "skip $f"; fi
f="$C/epoch=0100-val_loss=0.114.ckpt"
if [[ -f "$f" && $(stat -c %s "$f") == 3047655148 ]]; then rm "$f"; else echo "skip $f"; fi
f="$C/epoch=0120-val_loss=0.148.ckpt"   # also on geodude
if [[ -f "$f" && $(stat -c %s "$f") == 3047655148 ]]; then rm "$f"; else echo "skip $f"; fi
f="$C/epoch=0140-val_loss=0.181.ckpt"
if [[ -f "$f" && $(stat -c %s "$f") == 3047655148 ]]; then rm "$f"; else echo "skip $f"; fi
f="$C/epoch=0160-val_loss=0.201.ckpt"
if [[ -f "$f" && $(stat -c %s "$f") == 3047655148 ]]; then rm "$f"; else echo "skip $f"; fi
f="$C/epoch=0180-val_loss=0.234.ckpt"
if [[ -f "$f" && $(stat -c %s "$f") == 3047655148 ]]; then rm "$f"; else echo "skip $f"; fi
f="$C/epoch=0199-val_loss=0.243.ckpt"
if [[ -f "$f" && $(stat -c %s "$f") == 3047655148 ]]; then rm "$f"; else echo "skip $f"; fi

# 5.7G heading 43448962. e25 matches geodude at 3047598380 bytes. Job is finished.
d="$S/training_outputs/oct_v35_h4s_r4vt_heading_tilt_sc/oct-v35-h4s-r4vt-heading-tilt_sc_e200_bs64_valtopk_43448962"
if job_reads 43448962; then echo "skip $d"
elif [[ $(stat -c %s "$d/checkpoints/epoch=0025-val_loss=0.059.ckpt") == 3047598380 ]]; then
  rm -r "$d"
else echo "skip $d"; fi

# 5.7G track_clean. epoch 55 matches geodude at 3048139948 bytes.
d="$S/training_outputs/oct_v35_h4s_r4vt_track_clean/oct-v35-h4s-r4vt-track-clean_e100_bs64_valtopk_43351238"
if job_reads 43351238; then echo "skip $d"
elif [[ $(stat -c %s "$d/checkpoints/epoch=0055-val_loss=0.030.ckpt") == 3048139948 ]]; then
  rm -r "$d"
else echo "skip $d"; fi

# 5.7G step-mask. epoch 30 is 3048140140 bytes.
d="$S/training_outputs/oct_v35_h4s_r4vt_track_clean_step_mask/oct-v35-h4s-r4vt-track-clean-step-mask_e100_bs64_valtopk_43363017"
if job_reads 43363017; then echo "skip $d"
elif [[ $(stat -c %s "$d/checkpoints/epoch=0030-val_loss=0.033.ckpt") == 3048140140 ]]; then
  rm -r "$d"
else echo "skip $d"; fi

# 5.7G smooth frame. epoch 30 is 3048139948 bytes.
d="$S/training_outputs/oct_v35_h4s_r4vt_smooth_frame/oct-v35-h4s-r4vt-smooth_e100_bs64_valtopk_43335222"
if job_reads 43335222; then echo "skip $d"
elif [[ $(stat -c %s "$d/checkpoints/epoch=0030-val_loss=0.026.ckpt") == 3048139948 ]]; then
  rm -r "$d"
else echo "skip $d"; fi

# 5.7G frame continuity. epoch 30 is 3048140076 bytes.
d="$S/training_outputs/oct_v35_h4s_r4vt_frame_continuity_causal/oct-v35-h4s-r4vt-causal_e100_bs64_valtopk_43308311"
if job_reads 43308311; then echo "skip $d"
elif [[ $(stat -c %s "$d/checkpoints/epoch=0030-val_loss=0.032.ckpt") == 3048140076 ]]; then
  rm -r "$d"
else echo "skip $d"; fi

# 1.1M failed frame-continuity start.
d="$S/training_outputs/oct_v35_h4s_r4vt_frame_continuity_causal/oct-v35-h4s-r4vt-causal_e40_bs64_valtopk_43307939"
if job_reads 43307939; then echo "skip $d"; else
  rm -r "$d"
fi

# 5.7G epoch-0 robust-ab, superseded by running 43465423.
d="$S/training_outputs/oct_v35_h4s_r4vt_heading_tilt_sc_robustab/oct-v35-h4s-r4vt-heading-tilt_sc_robustab_e200_bs64_valtopk_43454035"
if job_reads 43454035; then echo "skip $d"
elif [[ $(stat -c %s "$d/checkpoints/epoch=0000-val_loss=0.244.ckpt") == 3047598572 ]]; then
  rm -r "$d"
else echo "skip $d"; fi

# ~1M failed tilt1 starts next to the live 43467749 run.
d="$S/training_outputs/oct_v35_h4s_r4vt_tilt1/oct-v35-h4s-r4vt-tilt1_e200_bs64_valtopk_43467751"
if job_reads 43467751; then echo "skip $d"; else rm -r "$d"; fi
d="$S/training_outputs/oct_v35_h4s_r4vt_tilt1/oct-v35-h4s-r4vt-tilt1_e200_bs64_valtopk_43467490"
if job_reads 43467490; then echo "skip $d"; else rm -r "$d"; fi

# ~3M failed fixedk3000 starts. The real runs are on weirdlab.
d="$S/training_outputs/r4vt_causal_foot_fixedk3000_cam/r4vt_causal_foot_fixedk3000_cam_43436610"
if job_reads 43436610; then echo "skip $d"; else rm -r "$d"; fi
d="$S/training_outputs/r4vt_causal_foot_fixedk3000_nocam/r4vt_causal_foot_fixedk3000_nocam_43436834"
if job_reads 43436834; then echo "skip $d"; else rm -r "$d"; fi
d="$S/training_outputs/r4vt_causal_foot_fixedk3000_nocam/r4vt_causal_foot_fixedk3000_nocam_43436862"
if job_reads 43436862; then echo "skip $d"; else rm -r "$d"; fi

# 3.34G tar, 3580805120 bytes, already extracted. The zarr stays.
tar="$S/umidata/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_causal_foot.tar"
zarr="$S/umidata/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_causal_foot/acp_replay_buffer.zarr"
if [[ -d "$zarr" && $(stat -c %s "$tar") == 3580805120 ]]; then
  rm "$tar"
else echo "skip $tar"; fi

U="$S/umidata/mold_aware_datasets"
# 5.8G fixedk3000 zarr. Live job reads carry_k3000, not this directory.
d="$U/stringer_v3.5_fixb_odom_corr/r4vt_causal_foot_fixedk3000"
if job_reads r4vt_causal_foot_fixedk3000; then echo "skip $d"; else rm -r "$d"; fi
# 5.8G world-fixb tilt zarr. No live sbatch names it.
d="$U/stringer_v3.5_fixb_odom_corr/r4vt_world_fixb_tilt"
if job_reads r4vt_world_fixb_tilt; then echo "skip $d"; else rm -r "$d"; fi
# 5.8G smooth tcp track.
d="$U/stringer_v3.5_fixb_odom_corr/r4vt_smooth_tcp_track_clean"
if job_reads r4vt_smooth_tcp_track_clean; then echo "skip $d"; else rm -r "$d"; fi
# 5.9G frame-continuity zarr.
d="$U/stringer_v3.5_fixb_odom_corr/r4vt_frame_continuity_causal"
if job_reads r4vt_frame_continuity_causal; then echo "skip $d"; else rm -r "$d"; fi
# 5.8G smooth-frame zarr.
d="$U/stringer_v3.5_fixb_odom_corr/r4vt_smooth_frame"
if job_reads r4vt_smooth_frame; then echo "skip $d"; else rm -r "$d"; fi
# 6.1G pre-odom_corr tree.
d="$U/stringer_v3.5_fixb"
if job_reads /stringer_v3.5_fixb/; then echo "skip $d"; else rm -r "$d"; fi
# 29M old scanner fixb.
d="$U/stringer_scanner_v2_fixb"
if job_reads stringer_scanner_v2_fixb; then echo "skip $d"; else rm -r "$d"; fi

# Stale scrubbed checkouts. Live jobs use UMI-FT-causal_foot and UMI-FT-tilt1.
d="$S/UMI-FT-step_mask"          # 275M
if job_reads UMI-FT-step_mask; then echo "skip $d"; else rm -r "$d"; fi
d="$S/UMI-FT-smooth_frame"       # 272M
if job_reads UMI-FT-smooth_frame; then echo "skip $d"; else rm -r "$d"; fi
d="$S/UMI-FT-frame_continuity"   # 269M
if job_reads UMI-FT-frame_continuity; then echo "skip $d"; else rm -r "$d"; fi
d="$S/UMI-FT-tilt_reweight"      # 180M
if job_reads UMI-FT-tilt_reweight; then echo "skip $d"; else rm -r "$d"; fi
d="$S/UMI-FT-surface_frames"     # 197M, older than the geodude checkout
if job_reads UMI-FT-surface_frames; then echo "skip $d"; else rm -r "$d"; fi

# 678M home cache. No job reads it.
if [[ -d "$HOME/.cache" ]]; then rm -r "$HOME/.cache"; fi

# Weirdlab. One probe file must match the geodude byte size before a directory goes.
# 41G each. October mold_aware, 2095201116 bytes, same as geodude.
d="$W/training_outputs/mold_aware_world_surface"
if job_reads mold_aware_world_surface; then echo "skip $d"
elif [[ $(stat -c %s "$d/vtevy/checkpoints/epoch=0000-train_loss=0.969.ckpt") == 2095201116 ]]; then
  rm -r "$d"
else echo "skip $d"; fi
d="$W/training_outputs/mold_aware_world_world"
if job_reads mold_aware_world_world; then echo "skip $d"
elif [[ $(stat -c %s "$d/vtevy/checkpoints/epoch=0000-train_loss=0.967.ckpt") == 2095201116 ]]; then
  rm -r "$d"
else echo "skip $d"; fi
d="$W/training_outputs/mold_aware_surface_surface"
if job_reads mold_aware_surface_surface; then echo "skip $d"
elif [[ $(stat -c %s "$d/vtevy/checkpoints/epoch=0000-train_loss=0.967.ckpt") == 2095201116 ]]; then
  rm -r "$d"
else echo "skip $d"; fi

# 7.9G each. Epoch-195 files that match geodude.
d="$W/training_outputs/oct_v35_fixb_r1_world_world/oct-v35-fixb-r1_world_world_e200_bs160_42581383"
if [[ $(stat -c %s "$d/checkpoints/epoch=0195-train_loss=0.013.ckpt") == 4190341037 ]]; then rm -r "$d"; else echo "skip $d"; fi
d="$W/training_outputs/oct_v35_fixb_r2_world_fixb/oct-v35-fixb-r2_world_fixb_e200_bs160_42747614"
if [[ $(stat -c %s "$d/checkpoints/epoch=0195-train_loss=0.013.ckpt") == 4190341101 ]]; then rm -r "$d"; else echo "skip $d"; fi
d="$W/training_outputs/oct_v35_fixb_r1rel_world/oct-v35-fixb-r1rel_world_e200_bs160_42747616"
if [[ $(stat -c %s "$d/checkpoints/epoch=0195-train_loss=0.014.ckpt") == 4190341421 ]]; then rm -r "$d"; else echo "skip $d"; fi
d="$W/training_outputs/oct_v35_fixb_r4_world_fixb_tilt/oct-v35-fixb-r4_world_fixb_tilt_e200_bs160_42747615"
if [[ $(stat -c %s "$d/checkpoints/epoch=0195-train_loss=0.013.ckpt") == 4190515885 ]]; then rm -r "$d"; else echo "skip $d"; fi
d="$W/training_outputs/oct_v35_fixb_r4_world_fixb_tilt/oct-v35-fixb-r4_world_fixb_tilt_e200_bs160_42581384"
if [[ $(stat -c %s "$d/checkpoints/epoch=0195-train_loss=0.013.ckpt") == 4190515629 ]]; then rm -r "$d"; else echo "skip $d"; fi

# 40G cam fixedk3000. Superseded by carry_k3000. Probe is 3047654380, geodude keeps e25 and latest.
d="$W/training_outputs/r4vt_causal_foot_fixedk3000_cam/r4vt_causal_foot_fixedk3000_cam_43437517"
if job_reads 43437517; then echo "skip $d"
elif [[ $(stat -c %s "$d/checkpoints/epoch=0025-val_loss=0.050.ckpt") == 3047654380 ]]; then
  rm -r "$d"
else echo "skip $d"; fi
# 20G nocam fixedk3000. Every epoch is 1496932127 bytes. Geodude keeps e25 and latest.
d="$W/training_outputs/r4vt_causal_foot_fixedk3000_nocam/r4vt_causal_foot_fixedk3000_nocam_43437525"
if job_reads 43437525; then echo "skip $d"
elif [[ $(stat -c %s "$d/checkpoints/epoch=0025-val_loss=0.050.ckpt") == 1496932127 ]]; then
  rm -r "$d"
else echo "skip $d"; fi

# 6.7G image, 7116107776 bytes, same as geodude v3.
f="$W/images/umift-train_v3.sif"
if [[ -f "$f" && $(stat -c %s "$f") == 7116107776 ]]; then rm "$f"; else echo "skip $f"; fi

# Stale weirdlab checkouts and old logs. Live sbatches are under scrubbed.
d="$W/UMI-FT"                              # 3.4G, mostly runs/
if job_reads /gscratch/weirdlab/vtevy/UMI-FT; then echo "skip $d"; else rm -r "$d"; fi
d="$W/UMI-FT-fixb.bak.20261029133219"      # 2.3G
if job_reads UMI-FT-fixb.bak; then echo "skip $d"; else rm -r "$d"; fi
d="$W/UMI-FT-fixb"                         # 357M
if job_reads /gscratch/weirdlab/vtevy/UMI-FT-fixb; then echo "skip $d"; else rm -r "$d"; fi
d="$W/logs"                                # 70M
if job_reads /gscratch/weirdlab/vtevy/logs; then echo "skip $d"; else rm -r "$d"; fi
