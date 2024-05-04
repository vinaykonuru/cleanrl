import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 1 get the results of the same algorithm on all the environments
# 2 do a min-max norm on each algorithms results
# 3 average the normalized results of each algorithm across all the envs

def extract_scalar_from_event_file(event_file):
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()  # load the events from the file

    # Retrieve the scalars you're interested in
    try:
        episodic_return = event_acc.Scalars('eval/training_avg')
    except:
        episodic_return = None
    return episodic_return

def find_event_files(alg_dir_path):
    event_file = ""

    for root, dirs, files in os.walk(alg_dir_path):
        for file in files:
            if file.startswith('events'):
                event_file = os.path.join(root, file)
                break
    return event_file

def get_subdirectories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def avg_untrained_return(scalars, learning_starts=100):
    training_step = 0
    count = 0
    avg = 0
    while training_step < learning_starts:
        training_step = scalars[count].step

        avg += scalars[count].value
        count += 1
    return avg / count

def normalization_values(env_name, learning_starts=30000):
    base_path = f"runs/{env_name}"  # replace with your actual path
    alg_folders = get_subdirectories(base_path)
    min_training_val = 0
    max_training_val = 0
    untrained_avg_returns_sum = 0

    for alg_folder in alg_folders:
        alg_folder_path = os.path.join(base_path, alg_folder)
        event_file = find_event_files(alg_folder_path)

        if os.path.exists(event_file):
            training_returns = extract_scalar_from_event_file(event_file)
        
        if training_returns is None:
            continue
        # get the average return of the untrained agents for all iterations
        untrained_avg_returns_sum += avg_untrained_return(training_returns)

        # get the max return of the trained agents
        if max(training_returns).value > max_training_val:
            max_training_val = max(training_returns).value

    min_training_val = untrained_avg_returns_sum / len(alg_folders)
    return min_training_val, max_training_val
