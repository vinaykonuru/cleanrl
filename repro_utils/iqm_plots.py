import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

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

def plot_single_iqms(results):
    fig, ax = plt.subplots()

    for key, data in results.items():
        ax.plot(data['step'], data['iqm'], label=key)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("IQM Episodic Returns")
    ax.legend()  # add a legend to distinguish the different lines

    plt.savefig('repro_utils/iqm_results/iqm_plot.png')

def main():
    env = "Hopper-v4"
    results = load_dict_from_file(f'repro_utils/iqm_results/results_{env}.json')
    plot_single_iqms(results)

if __name__ == "__main__":
    main()