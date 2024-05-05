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
#SBATCH --mail-user=vkonuru@princeton.edu

module purge
module load anaconda3/2023.9
conda activate cleanRL

python tests/test_doublegum.py --num-trials 3 --double_layer --total-timesteps 500000 --learning-starts 30000 --env_id Walker2d-v4
# python cleanrl/td3_continuous_action.py --seed 168 --double_layer --save-model --env-id Ant-v4 --learning_starts 30000 --total-timesteps 500000 --save-model-folder Ant-v4_Group_Norm
# python cleanrl/td3_continuous_action.py --seed 168 --double_layer --save-model --env-id Ant-v4 --learning_starts 30000 --total-timesteps 500000 --save-model-folder Ant-v4_Group_Norm

# python cleanrl/td3_continuous_action.py --seed 168 --double_layer --save-model --env-id Ant-v4 --learning_starts 30000 --total-timesteps 500000 --save-model-folder Ant-v4_Group_Norm
# python cleanrl/td3_continuous_action.py --seed 493 --double_layer --save-model --env-id Ant-v4 --learning_starts 30000 --total-timesteps 500000 --save-model-folder Ant-v4_Group_Norm
# python cleanrl/td3_continuous_action.py --seed 527 --double_layer --save-model --env-id Ant-v4 --learning_starts 30000 --total-timesteps 500000 --save-model-folder Ant-v4_Group_Norm
# python cleanrl/td3_continuous_action.py --seed 534 --double_layer --save-model --env-id Ant-v4 --learning_starts 30000 --total-timesteps 500000 --save-model-folder Ant-v4_Group_Norm
# python cleanrl/td3_continuous_action.py --seed 584 --double_layer --save-model --env-id Ant-v4 --learning_starts 30000 --total-timesteps 500000 --save-model-folder Ant-v4_Group_Norm

# python cleanrl/td3_continuous_action.py --seed 168 --double_layer --save-model --env-id HalfCheetah-v4 --learning_starts 30000 --total-timesteps 500000 --save-model-folder HalfCheetah-v4_Group_Norm
# python cleanrl/td3_continuous_action.py --seed 493 --double_layer --save-model --env-id HalfCheetah-v4 --learning_starts 30000 --total-timesteps 500000 --save-model-folder HalfCheetah-v4_Group_Norm
# python cleanrl/td3_continuous_action.py --seed 527 --double_layer --save-model --env-id HalfCheetah-v4 --learning_starts 30000 --total-timesteps 500000 --save-model-folder HalfCheetah-v4_Group_Norm
# python cleanrl/td3_continuous_action.py --seed 534 --double_layer --save-model --env-id HalfCheetah-v4 --learning_starts 30000 --total-timesteps 500000 --save-model-folder HalfCheetah-v4_Group_Norm
# python cleanrl/td3_continuous_action.py --seed 584 --double_layer --save-model --env-id HalfCheetah-v4 --learning_starts 30000 --total-timesteps 500000 --save-model-folder HalfCheetah-v4_Group_Norm