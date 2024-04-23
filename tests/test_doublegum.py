import subprocess


def test_mujoco():
    """
    Test mujoco
    """
    subprocess.run(
        "python cleanrl/ddpg_continuous_action.py --env-id Hopper-v4 --learning-starts 100 --batch-size 32 --total-timesteps 100000",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python cleanrl/dgum_continuous_action.py --env-id Hopper-v4 --learning-starts 100 --batch-size 32 --total-timesteps 100000",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python cleanrl/td3_continuous_action.py --env-id Hopper-v4 --learning-starts 100 --batch-size 32 --total-timesteps 100000",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python cleanrl/sac_continuous_action.py --env-id Hopper-v4 --learning-starts 100 --batch-size 32 --total-timesteps 100000",
        shell=True,
        check=True,
    )
def test_mujoco_eval():
    """
    Test mujoco_eval
    """

    try:
        subprocess.run(
            "python cleanrl/dgum_continuous_action.py --save-model --env-id Hopper-v4 --learning-starts 30000 --batch-size 256 --total-timesteps 100000",
            shell=True,
            check=True,
        )
    except Exception as e:
        print(e)

    try:    
        subprocess.run(
            "python cleanrl/ddpg_continuous_action.py --save-model --env-id Hopper-v4 --learning-starts 30000 --batch-size 256 --total-timesteps 100000",
            shell=True,
            check=True,
        )
    except Exception as e:
        print(e)

    try:
        subprocess.run(
            "python cleanrl/td3_continuous_action.py --save-model --env-id Hopper-v4 --learning-starts 30000 --batch-size 256 --total-timesteps 100000",
            shell=True,
            check=True,
        )
    except Exception as e:
        print(e)
    
    try:
        subprocess.run(
            "python cleanrl/sac_continuous_action.py --save-model --env-id Hopper-v4 --learning-starts 30000 --batch-size 256 --total-timesteps 100000",
            shell=True,
            check=True,
        )
    except Exception as e:
        print(e)
    
if __name__ == "__main__":
    test_mujoco_eval()