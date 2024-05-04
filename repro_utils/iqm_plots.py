import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
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

def plot_single_iqms(steps, results, smoothing_weight):
    fig, ax = plt.subplots()

    for key, values in results.items():
        ax.plot(steps, values, label=key)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("IQM Episodic Returns")
    ax.set_title(f"Evaluation with smoothing weight: {smoothing_weight}")
    ax.legend()  # add a legend to distinguish the different lines

    plt.savefig('repro_utils/iqm_results/iqm_plot.png')

def main():
    algs = ['dgum', 'ddpg', 'sac']
    envs = ['Hopper-v4_Group_Norm', 'Walker2d-v4_Group_Norm']
    smoothing_weight = 0.9
    env_results = []
    for env in envs:
        env_results.append(load_dict_from_file(f'repro_utils/iqm_results/results_{env}.json'))
    
    # env_results = calculate_env_iqms(algs, envs, smoothing_weight)

    steps = env_results[0]['dgum']['step']
    total_vals = {}

    total_vals['dgum'] = np.zeros(len(steps))
    total_vals['ddpg'] = np.zeros(len(steps))
    total_vals['sac'] = np.zeros(len(steps))
    total_vals['td3'] = np.zeros(len(steps))

    for env in env_results:
        for key, data in env.items():
            total_vals[key] += np.array(data['iqm'])
    
    total_vals = {key: value / len(env_results) for key, value in total_vals.items()}
    
    plot_single_iqms(steps, total_vals, smoothing_weight)

if __name__ == "__main__":
    main()