#!/usr/bin/env bash
# DHALF reproduction launcher.
#
# Usage:
#   ./run.sh 151673
#   ./run.sh 151674
#   ./run.sh E18_mouse_brain
#   ./run.sh Stereo_AB
#   ./run.sh BRCA
#
# Environment overrides:
#   PYTHON            Python interpreter to use              (default: python)
#   GPU               CUDA_VISIBLE_DEVICES                   (default: 0)
#   SEED              base seed; trial i uses SEED + i       (default: 0)
#   NTRIALS           number of trials                       (default: 10)
#   EPOCHS            training epochs                        (default: dataset table below)
#   EVAL_FREQ         evaluation interval                    (default: dataset table below)
#   RESULT_NAME       output directory / filename prefix     (default: <dataset>)
#   EVAL_AUX          1 also scores z_full / concat          (default: 1)
#   SAVE_DIAGNOSTICS  1 also dumps gate + branch diagnostics (default: 1)
#
# Examples:
#   PYTHON=/opt/conda/envs/dhalf/bin/python GPU=1 ./run.sh 151673
#   NTRIALS=1 EPOCHS=5 EVAL_FREQ=5 ./run.sh BRCA       # quick check
#
# Data files are NOT shipped with this repository; see the "Data" section of
# README.md for the expected directory layout and download links.

set -euo pipefail

cd "$(dirname "$0")"

DATASET="${1:-}"
if [ -z "$DATASET" ]; then
  echo "usage: $0 <151673|151674|E18_mouse_brain|Stereo_AB|BRCA>" >&2
  exit 2
fi

# Hyperparameters below follow the paper's dataset-specific table
# (spatial neighbours k, refinement neighbours k_r, discriminator temperature,
# homophilic / heterophilic margins, gate width, semantic neighbours k_s,
# loss weights, epochs and evaluation interval).
case "$DATASET" in
  151673)
    KEY=151673; DATA_READY="data/151673/filtered_feature_bc_matrix.h5"
    K=40; REFINE_K=25; TEMP=1.0; M_HOM=0.4; M_HET=0.6
    GATE_HIDDEN=64; SEM_K=10
    L_BASE=0.0; L_FL=1.0; L_FH=1.0; L_MIX=1.0
    DEF_EPOCHS=600; DEF_EVAL=20
    ;;
  151674)
    KEY=151674; DATA_READY="data/151674/filtered_feature_bc_matrix.h5"
    K=40; REFINE_K=25; TEMP=1.0; M_HOM=0.4; M_HET=0.6
    GATE_HIDDEN=32; SEM_K=5
    L_BASE=0.1; L_FL=1.0; L_FH=0.5; L_MIX=1.0
    DEF_EPOCHS=50; DEF_EVAL=5
    ;;
  E18_mouse_brain|E18.5|E18)
    DATASET=E18_mouse_brain
    KEY=E18_mouse_brain; DATA_READY="data/E18.5_mouse_brain/E18_adata_rna.h5ad"
    K=30; REFINE_K=10; TEMP=1.0; M_HOM=0.3; M_HET=0.7
    GATE_HIDDEN=32; SEM_K=5
    L_BASE=0.1; L_FL=1.0; L_FH=0.75; L_MIX=1.0
    DEF_EPOCHS=600; DEF_EVAL=20
    ;;
  Stereo_AB)
    KEY=Stereo_AB; DATA_READY="data/Stereo/Stereo_AB.h5ad"
    K=20; REFINE_K=15; TEMP=1.0; M_HOM=0.4; M_HET=0.6
    GATE_HIDDEN=32; SEM_K=5
    L_BASE=0.1; L_FL=1.0; L_FH=0.25; L_MIX=1.0
    DEF_EPOCHS=50; DEF_EVAL=5
    ;;
  BRCA|Breast_Caner)
    DATASET=BRCA
    KEY=BRCA; DATA_READY="data/BRCA/Breast_Caner/filtered_feature_bc_matrix.h5"
    K=8; REFINE_K=5; TEMP=0.7; M_HOM=0.6; M_HET=0.4
    GATE_HIDDEN=64; SEM_K=10
    L_BASE=0.0; L_FL=1.0; L_FH=0.0; L_MIX=1.0
    DEF_EPOCHS=600; DEF_EVAL=20
    ;;
  *)
    echo "unknown dataset: $DATASET" >&2
    echo "choose from: 151673, 151674, E18_mouse_brain, Stereo_AB, BRCA" >&2
    exit 2
    ;;
esac

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
NTRIALS="${NTRIALS:-10}"
EPOCHS="${EPOCHS:-$DEF_EPOCHS}"
EVAL_FREQ="${EVAL_FREQ:-$DEF_EVAL}"
RESULT_NAME="${RESULT_NAME:-$KEY}"
EVAL_AUX="${EVAL_AUX:-1}"
SAVE_DIAGNOSTICS="${SAVE_DIAGNOSTICS:-1}"

if [ ! -e "$DATA_READY" ]; then
  echo "error: missing $DATA_READY" >&2
  echo "       download the data first, see the 'Data' section of README.md" >&2
  exit 1
fi

echo "[DHALF] dataset=$DATASET trials=$NTRIALS epochs=$EPOCHS eval_freq=$EVAL_FREQ gpu=$GPU"

"$PYTHON" main.py \
  -dataset "$DATASET" \
  -gpu "$GPU" \
  --seed "$SEED" \
  -ntrials "$NTRIALS" \
  -k "$K" \
  -refine_k "$REFINE_K" \
  --temp_disc "$TEMP" \
  -margin_hom "$M_HOM" \
  -margin_het "$M_HET" \
  -alpha 0.1 \
  -nlayers_enc 2 \
  -nlayers_proj 1 \
  -emb_dim 512 \
  -proj_dim 512 \
  -dropout 0.5 \
  -cl_rounds 2 \
  -epochs "$EPOCHS" \
  -eval_freq "$EVAL_FREQ" \
  --sem_k "$SEM_K" \
  --lambda_fl "$L_FL" \
  --lambda_fh "$L_FH" \
  --lambda_base "$L_BASE" \
  --lambda_mix "$L_MIX" \
  --mix_detach_branches 1 \
  --variant_d_warmup 0 \
  --variant_d_gate_mode node \
  --gate_hidden "$GATE_HIDDEN" \
  --full_mlp_hidden 256 \
  --full_mlp_out 512 \
  --use_hom_pos 1 \
  -lr_gcl 0.0001 \
  -lr_disc 0.0001 \
  -w_decay 0.0 \
  --eval_aux_embeddings "$EVAL_AUX" \
  --save_diagnostics "$SAVE_DIAGNOSTICS" \
  --result_name "$RESULT_NAME"
