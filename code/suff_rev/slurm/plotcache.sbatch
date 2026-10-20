#!/bin/bash
# Rebuild the lost plot caches suff_gbc_plotdata.npz and suff_p2_plotdata.npz (round 3) by running
# code/suff/suff_gbc.py and suff_p2.py unchanged in the Hopper copy under $BASE; figures go to
# $BASE/figs (not used for the book). Results are copied back to code/suff_rev/plotcache/.
#SBATCH --job-name=smg_suff_cache
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/cache_%A_%a.log
BASE=${SMG_BASE:-/scratch/$USER/smg_suff_r2}
cd $BASE
S=(suff_gbc suff_p2)
export OMP_NUM_THREADS=2 SMG_FIG_DIR=$BASE/figs
mkdir -p $BASE/figs
$BASE/venv/bin/python -u -I code/suff/${S[$SLURM_ARRAY_TASK_ID]}.py
