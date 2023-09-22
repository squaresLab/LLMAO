from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import os
import json
import argparse


def roc_plotter(label_name, labels, probabilities):
    fpr, tpr, thresholds  = roc_curve(labels, probabilities)
    # print(f'{label_name} AUC: {round(roc_auc_score(labels, probabilities), 3)}')

    for i, threshold in enumerate(thresholds):
        if round(threshold,3) == 0.01 and 'Devign-16B' in label_name:
            print('false positive ', round(fpr[i],2))
            print('true positive ', round(tpr[i],2))
            print('threshold ', threshold)
            break

    if '16B' in label_name:
        plt.plot(fpr, tpr, linestyle='--',
                 label=label_name, color='red')
    elif '6B' in label_name:
        plt.plot(fpr, tpr, linestyle='--',
                 label=label_name, color='orange')
    elif '350M' in label_name:
        plt.plot(fpr, tpr, linestyle='--',
                 label=label_name, color='blue')
    elif 'scratch' in label_name:
        label_here = label_name.replace('scratch', 'from-scratch')
        plt.plot(fpr, tpr, linestyle='--',
                 label=label_here, color='green')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


def results_plot(log_path):
    data_list = ['bugsinpy', 'defects4j', 'devign']
    data_list = ['devign']
    for data_name in data_list:
        plt.axline((0, 0), slope=1, color='black', label='Random')
        for subdir, _, files in os.walk(log_path):
            for file in files:
                if '.json' in file:
                    if data_name not in subdir:
                        continue
                    f = open(os.path.join(subdir, file))
                    split_dir = subdir.split('_')
                    data_name = split_dir[0]
                    data_name = data_name.split('/')[-1]
                    params = split_dir[1]
                    if params == '256' or params == '1024':
                        params = 'from_scratch'
                    data = json.load(f)
                    probabilities = data['prob']
                    labels = data["label"]
                    f.close()
                    filtered_prob = []
                    filtered_label = []
                    for i, prob in enumerate(probabilities):
                        if prob != 0:
                            filtered_prob.append(prob)
                            filtered_label.append(labels[i])
                    label_name = f'{data_name}-{params}'.replace('--', '-').replace(
                        'bugsinpy', 'BugsInPy').replace('defects4j', 'Defects4J').replace('devign', 'Devign')
                    roc_plotter(label_name, filtered_label, filtered_prob)

        handles, labels = plt.gca().get_legend_handles_labels()
        if 'Devign' in label_name:
            order = [0, 1, 2, 4, 3]
        elif 'Defects4J' in label_name:
            order = [0, 1, 3, 2, 4]
        elif 'BugsInPy' in label_name:
            order = [0, 3, 2, 1, 4]
        plt.legend([handles[idx] for idx in order], [labels[idx]
                for idx in order], loc='lower right')
        plt.savefig(os.path.join('plots/', f'{data_name}_roc.pdf'))
        plt.clf()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", help="Path to data root")
    args = ap.parse_args()
    log_path = args.log_path
    results_plot(log_path)
