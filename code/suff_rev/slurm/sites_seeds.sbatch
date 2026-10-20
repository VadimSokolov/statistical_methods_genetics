#!/bin/bash
# Seed study for Table suff:tab:sites (item suff_b-08). Layout on Hopper under $BASE:
#   book.tex (empty marker), code/suff/{gbcnet.py,suff_p2.py}, code/suff_rev/sites_seeds.py,
#   venv with jax[cpu] and optax (python -m venv --system-site-packages venv; pip install "jax[cpu]" optax).
# Task i fits config CONFIGS[i / 5] for seeds 10 (i mod 5) to 10 (i mod 5) + 9, so 50 seeds per config.
#SBATCH --job-name=smg_suff_seeds
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/seeds_%A_%a.log
BASE=${SMG_BASE:-/scratch/$USER/smg_suff_r2}
cd $BASE
CONFIGS=(S_only d1 d2 d4)
i=${SLURM_ARRAY_TASK_ID}
CONFIG=${CONFIGS[$((i / 5))]}
SEED0=$(( (i % 5) * 10 ))
NSEED=${NSEED:-10}
export OMP_NUM_THREADS=2
$BASE/venv/bin/python -u -I code/suff_rev/sites_seeds.py fit $CONFIG $SEED0 $NSEED $BASE/results
