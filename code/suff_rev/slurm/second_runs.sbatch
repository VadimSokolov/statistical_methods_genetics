#!/bin/bash
# Second set of runs (capacity, calibration, location scale) for chapters/suff.tex, round 3.
# Same layout on Hopper as sites_seeds.sbatch, under $BASE, plus code/suff/suff_gbc.py and
# code/suff_rev/second_runs.py. Task i fits config CONFIGS[i / 4] for seeds 5 (i mod 4) to
# 5 (i mod 4) + 4, so 20 seeds per config.
#SBATCH --job-name=smg_suff_r3
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-31
#SBATCH --output=logs/r3_%A_%a.log
BASE=${SMG_BASE:-/scratch/$USER/smg_suff_r2}
cd $BASE
CONFIGS=(d1_h64_s700 d1_h128_s700 d1_h256_s700 d4_h64_s700 d8_h64_s700 d1_h64_s2100 tail_s2100 ls_s2100)
i=${SLURM_ARRAY_TASK_ID}
CONFIG=${CONFIGS[$((i / 4))]}
SEED0=$(( (i % 4) * 5 ))
NSEED=${NSEED:-5}
export OMP_NUM_THREADS=2
$BASE/venv/bin/python -u -I code/suff_rev/second_runs.py fit $CONFIG $SEED0 $NSEED $BASE/results_r3
