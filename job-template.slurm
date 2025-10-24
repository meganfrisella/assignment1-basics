#!/bin/bash
#SBATCH --job-name=pytest-job             # Job name

#SBATCH --account=mfris                   # Replace with your netid
#SBATCH --partition=gpu                   # Partition name (default is gpu)
#SBATCH --nodes=1                         # One node (tomago/tempura)
#SBATCH --ntasks-per-node=1               # Number of CPU cores (vary it as needed)
#SBATCH --mem=16G                         # Memory per node
#SBATCH --gpus=1                          # Number of GPUs (for HW1, let it be 1)
#SBATCH --time=02:00:00                   # Max runtime = 2 hours (cluster limit)

#SBATCH --chdir=/homes/iws/mfris/assignment1-basics  # Working directory
#SBATCH --export=ALL                      # Export environment variables
#SBATCH --output=mfris-slurm-%j.out             # STDOUT log file (%j = JobID)
#SBATCH --error=mfris-slurm-%j.err              # STDERR log file (%j = JobID)

# --- Example Command to run ---
srun uv run src/train.py \
    --data-path data/tokenized-TinyStoriesV2-GPT4-valid.npy \
    --epochs 4 \
    --batch-size 32 \
    --context-length 256 \
    --vocab-size 50257 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --rope-theta 10000.0 \
    --epochs 100 \
    --lr 1e-3 \