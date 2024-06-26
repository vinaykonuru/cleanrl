import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
from glob import glob
from pathlib import Path
import os
import numpy as np
from env_normalization import normalization_values
import json
from typing import List

def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def load_tensorboard_events(logdir, smoothing_weight=0.9):
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()

    # Getting all the data from the TensorBoard file
    scalar_tags = event_acc.Tags()['scalars']

    data = {}
    for tag in scalar_tags:
        # Get the data for each tag
        events = event_acc.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        data[tag] = {'steps': steps, 'values': values}
        data[tag]['values'] = smooth(data[tag]['values'], smoothing_weight)
    return data

# Calculates IQM for all algos on an env
# make sure that all the iterations of a given algorithm are trained for the same number of steps
def get_iqm(sub_folder_path, algos):

    sf_path = Path(sub_folder_path)

    env_id = sf_path.name

    result_dict = {}

    for algo in algos:
        event_paths = sorted(glob(os.path.join(sf_path,f"*{algo}_cont*","events.out*")))
        print(event_paths)

        # if no files found then just continue
        if len(event_paths) == 0:
            continue

        steps = None
        ep_returns = [] # list of list of returns
        iqm = []
        
        for ev_path in event_paths:
            try:
                data = load_tensorboard_events(ev_path)['eval/training_avg']
            except:
                continue
            steps = data['steps']

            if len(data['values']) != 500:
                continue
            ep_returns.append(data['values'])
        ep_returns = np.array(ep_returns).T

        

        for step in ep_returns:
            iqm.append(interquartile_mean(step))

        
        result_dict[algo] = {
            "step": steps,
            "iqm": iqm
        }
    
    return result_dict

def interquartile_mean(arr):
    # Sort the array
    sorted_arr = np.sort(arr)
    
    # Get the length of the array
    arr_len = len(sorted_arr)
    
    # Calculate the start and end indices for the middle half values
    start_index = arr_len // 4
    end_index = 3 * arr_len // 4
    
    # Get the middle half values
    middle_half = sorted_arr[start_index:end_index]
    
    return middle_half.mean()

def plot_iqms(results):
    for key, data in results.items():
        fig, ax = plt.subplots()

        ax.plot(data['step'],data['iqm'])

        ax.set_xlabel("Training Step")
        ax.set_ylabel("IQM Episodic Returns")
        ax.set_title(f"Evaluation for {key}")

        fig.show()


def save_dict_to_file(dict_obj, file_name):
    with open(file_name, 'w') as file:
        json.dump(dict_obj, file)


def calculate_env_iqms(algs, envs, smoothing_weight):
    env_results = []
    for env in envs:
        results = get_iqm(f"runs/{env}", algs)
        min, max = normalization_values(env)
        for alg in algs:
            print(results[alg]["iqm"])
            results[alg]["iqm"] = [(value - min) / (max - min) for value in results[alg]["iqm"]]
        save_dict_to_file(results, f'repro_utils/iqm_results/results_{env}.json')
        env_results.append(results)
    return env_results

