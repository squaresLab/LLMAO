from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import os
import json
import argparse
import numpy as np


def hit_ratio(hit_counter, label_bug_counter):
    return round(hit_counter/label_bug_counter * 100, 1)


def hit_scores(probabilities, labels, proj_split, bug_split):
    prob_cutoffs = [0.3, 0.45, 0.65]
    n_tops = [5, 3, 1]

    for i, prob_cutoff in enumerate(prob_cutoffs):
        n_top = n_tops[i]
        return_counter = 0
        label_bug_counter = 0
        predicted_bug_counter = 0
        hit_counter = 0
        split_probs_proj = np.array_split(probabilities, proj_split)
        split_labels_proj = np.array_split(labels, proj_split)
        for proj_idx in range(len(split_probs_proj)):
            prob_project = split_probs_proj[proj_idx]
            label_project = split_labels_proj[proj_idx]
            split_probs_bugs = np.array_split(prob_project, bug_split)
            split_labels_bugs = np.array_split(label_project, bug_split)
            for bug_idx in range(len(split_probs_bugs)):
                prob_bug = split_probs_bugs[bug_idx]
                label_bug = split_labels_bugs[bug_idx]
                label_bug_counter += sum(label_bug)
                prob_bug = list(
                    map(lambda x: 1. if x > prob_cutoff else 0., prob_bug))
                predicted_bug_counter += sum(prob_bug)
                intersection = [prob_bug.index(n) for m, n in zip(
                    prob_bug, label_bug) if (n == m and n == 1.0)]
                correct_preds = len(intersection)
                if correct_preds > 0:
                    hit_counter += 1
        label_bug_counter = round(label_bug_counter / bug_split)

        print(f'Top {n_top}: {hit_counter}/{label_bug_counter} - {round(hit_counter/label_bug_counter * 100,1)}%')
        if n_top == 5:
            top_5 = hit_counter
        elif n_top == 3:
            top_3 = hit_counter
        elif n_top == 1:
            top_1 = hit_counter
        return_counter = label_bug_counter

        label_bug_counter = 0
    return return_counter, top_5, top_3, top_1


def roc_plotter(label_name, dimension, labels, probabilities):
    fpr, tpr, _ = roc_curve(labels, probabilities)
    if '16B' in label_name:
        plt.plot(fpr, tpr, linestyle='--',
                 label=label_name, color='red')
    elif '6B' in label_name:
        plt.plot(fpr, tpr, linestyle='--',
                 label=label_name, color='orange')
    elif '350M' in label_name:
        plt.plot(fpr, tpr, linestyle='--',
                 label=label_name, color='blue')
    elif 'Transformer' in label_name:
        label_here = label_name.replace('Transformer', 'from-scratch')
        plt.plot(fpr, tpr, linestyle='--',
                 label=label_here, color='green')

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


def results_plot(data_path):
    plotting = True
    print('---------------------')
    data_list = ['bugsinpy', 'defects4j', 'devign']
    data_list = ['defects4j']
    for data_name in data_list:
        plt.axline((0, 0), slope=1, color='black', label='Random')
        os.chdir(data_path)
        total_bugs = 0
        total_top_5 = 0
        total_top_3 = 0
        total_top_1 = 0
        for subdir, _, files in os.walk('model_logs'):
            for file in files:
                if '.json' in file:
                    if data_name not in subdir:
                        continue
                    f = open(os.path.join(subdir, file))
                    split_dir = subdir.split('_')
                    data_name = split_dir[1].replace('logs/', '')
                    params = split_dir[2]
                    if params == '256' or params == '1024':
                        params = 'Transformer'
                    dimension = split_dir[3]
                    if len(split_dir) > 4:
                        cross_fold_number = split_dir[4]
                    if 'scratch' in dimension:
                        dimension = ''
                    step_count = file.split('_')[1].replace('.json', '')
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
                    
                    print(label_name)

                    if 'BugsInPy' in label_name:
                        proj_split = 5
                        bug_split = 10
                    elif 'Defects4J' in label_name:
                        proj_split = 3
                        bug_split = 14
                    elif 'Devign' in label_name:
                        proj_split = 15
                        bug_split = 18

                    top_score_label_name = "Defects4J-16B"
                    top_score_dimension = "512"
                    if top_score_label_name in label_name and top_score_dimension in dimension:
                        print('hi')
                        print(
                            f'\nHit results for {label_name}-{dimension}-{cross_fold_number}\n')
                        return_counter, top_5, top_3, top_1 = hit_scores(
                            probabilities, labels, proj_split, bug_split)
                        total_bugs += return_counter
                        total_top_5 += top_5
                        total_top_3 += top_3
                        total_top_1 += top_1


                    if plotting:
                        roc_plotter(label_name, dimension, filtered_label, filtered_prob)

        total_bugs = 395
        print(
            f'Top 1,3,5 of {total_bugs} total bugs: [{total_top_1}({hit_ratio(total_top_1,total_bugs)}\%) & {total_top_3}({hit_ratio(total_top_3,total_bugs)}\%) & {total_top_5}({hit_ratio(total_top_5,total_bugs)}\%)]')

        
        if plotting:
            handles, labels = plt.gca().get_legend_handles_labels()
            print(len(handles))
            if 'Devign' in label_name:
                order = [0, 4, 2, 1, 3]
            elif 'Defects4J' in label_name:
                order = [0, 3, 1, 4, 2]
            elif 'BugsInPy' in label_name:
                order = [0, 2, 3, 1, 4]
            plt.legend([handles[idx] for idx in order], [labels[idx]
                    for idx in order], loc='lower right')

        print('\n')
        print('---------------------')
        plt.savefig(os.path.join(
            data_path + '/plots/', f'{data_name}_roc.pdf'))
        plt.clf()
        


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path", help="Path to data root")
    args = ap.parse_args()
    data_path = args.data_path
    results_plot(data_path)
