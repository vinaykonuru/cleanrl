#!/bin/bash
#SBATCH --job-name=vkonuru_cleanrl     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1            # number of GPUs per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=rswamy@princeton.edu

module purge
module load anaconda3/2023.9
conda activate cleanrl

# python tests/test_doublegum.py --num-trials 8 --double_layer --total-timesteps 500000 --learning-starts 30000 --env_id HalfCheetah-v4
python tests/test_doublegum_pessimism.py --num-trials 1 --double_layer --total-timesteps 500000 --learning-starts 30000 --env_id Hopper-v4