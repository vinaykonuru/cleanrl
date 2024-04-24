import subprocess
import tyro
import os
from dataclasses import dataclass
import numpy as np


@dataclass
class Args:
    num_trials: int = 1
    """number of trials for each algorithm"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Walker2d-v4"
    """the environment id of the Atari game"""
    total_timesteps: int = 600000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    pessimism_factor: float = -0.5
    """ pessimism factor for the target Q """

def test_mujoco_eval(args):
    """
    Test mujoco_eval
    """

    np.random.seed(args.seed)

    for _ in range(args.num_trials):

        rand_int = np.random.randint(low=0, high=1000)

        try:
            subprocess.run(
                f"python cleanrl/dgum_continuous_action.py --seed {rand_int} --save-model --env-id {args.env_id} --learning-starts {args.learning_starts} --batch-size 256 --total-timesteps {args.total_timesteps}",
                shell=True,
                check=True,
            )
        except Exception as e:
            print(e)

        try:    
            subprocess.run(
                f"python cleanrl/ddpg_continuous_action.py --seed {rand_int} --save-model --env-id {args.env_id} --learning-starts {args.learning_starts} --batch-size 256 --total-timesteps {args.total_timesteps}",
                shell=True,
                check=True,
            )
        except Exception as e:
            print(e)

        try:
            subprocess.run(
                f"python cleanrl/td3_continuous_action.py --seed {rand_int} --save-model --env-id {args.env_id} --learning-starts {args.learning_starts} --batch-size 256 --total-timesteps {args.total_timesteps}",
                shell=True,
                check=True,
            )
        except Exception as e:
            print(e)
        
        try:
            subprocess.run(
                f"python cleanrl/sac_continuous_action.py --seed {rand_int} --save-model --env-id {args.env_id} --learning-starts {args.learning_starts} --batch-size 256 --total-timesteps {args.total_timesteps}",
                shell=True,
                check=True,
            )
        except Exception as e:
            print(e)
    
if __name__ == "__main__":
    args = tyro.cli(Args)
    test_mujoco_eval(args)