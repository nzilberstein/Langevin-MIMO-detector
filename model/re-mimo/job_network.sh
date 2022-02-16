#!/bin/bash
#SBATCH --job-name=eval_full_corr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=kr.pratik73@gmail.com

#module purge

module load pre2019
module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176
module load matplotlib/2.1.1-foss-2017b-Python-3.6.3

export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH

srun python3 test_network.py
