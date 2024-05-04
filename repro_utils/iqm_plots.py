import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from iqm import calculate_env_iqms

def load_dict_from_file(file_name):
    with open(file_name, 'r') as file:
        return json.load(file)

def plot_iqms(results):
    for key, data in results.items():
        fig, ax = plt.subplots()

        ax.plot(data['step'],data['iqm'])

        ax.set_xlabel("Training Step")
        ax.set_ylabel("IQM Episodic Returns")
        ax.set_title(f"Evaluation for {key}")

    plt.savefig(f'repro_utils/iqm_results/iqm_plots_{key}.png')

def plot_single_iqms(results, env, smoothing_weight):
    fig, ax = plt.subplots()

    for key, data in results.items():
        ax.plot(data['step'], data['iqm'], label=key)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("IQM Episodic Returns")
    ax.set_title(f"Evaluation for Environment: {env} and smoothing weight: {smoothing_weight}")
    ax.legend()  # add a legend to distinguish the different lines

    plt.savefig('repro_utils/iqm_results/iqm_plot.png')

def main():
    algs = ['dgum', 'ddpg', 'sac', 'td3']
    envs = ['Hopper-v4']
    smoothing_weight = 0.9
    env_results = []
    for env in envs:
        env_results.append(load_dict_from_file(f'repro_utils/iqm_results/results_{env}.json'))
    
    # env_results = calculate_env_iqms(algs, envs, smoothing_weight)
    for count in range(len(env_results)):
        plot_single_iqms(env_results[count], envs[count], smoothing_weight)

if __name__ == "__main__":
    main()