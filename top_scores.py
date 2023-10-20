import os
import json
import argparse
import numpy as np


def top_ratio(hit_counter, label_bug_counter):
    return round(hit_counter / label_bug_counter * 100, 1)


def top_scores(probabilities, labels, label_name):
    n_tops = [5, 3, 1]
    if "bugsinpy" in label_name:
        data_split = 15
        window_split = 15
        prob_cutoffs = [0.1, 0.2, 0.3]
    elif "defects4j" in label_name:
        data_split = 15
        window_split = 14
        prob_cutoffs = [0.05, 0.1, 0.2]
    elif "devign" in label_name:
        data_split = 20
        window_split = 50
        prob_cutoffs = [0.35, 0.5, 0.6]

    for i, prob_cutoff in enumerate(prob_cutoffs):
        n_top = n_tops[i]
        label_bug_counter = 0
        predicted_bug_counter = 0
        hit_counter = 0
        split_probs_proj = np.array_split(probabilities, data_split)
        split_labels_proj = np.array_split(labels, data_split)
        for proj_idx in range(len(split_probs_proj)):
            prob_project = split_probs_proj[proj_idx]
            label_project = split_labels_proj[proj_idx]
            split_probs_bugs = np.array_split(prob_project, window_split)
            split_labels_bugs = np.array_split(label_project, window_split)
            for bug_idx in range(len(split_probs_bugs)):
                prob_bug = split_probs_bugs[bug_idx]
                label_bug = split_labels_bugs[bug_idx]
                label_bug_counter += sum(label_bug)
                prob_bug = list(
                    map(lambda x: 1.0 if x > prob_cutoff else 0.0, prob_bug)
                )
                predicted_bug_counter += sum(prob_bug)
                intersection = [
                    prob_bug.index(n)
                    for m, n in zip(prob_bug, label_bug)
                    if (n == m and n == 1.0)
                ]
                correct_preds = len(intersection)
                if correct_preds > 0:
                    hit_counter += 1
        label_bug_counter = round(label_bug_counter / window_split)
        if n_top == 5:
            top_5 = hit_counter
        elif n_top == 3:
            top_3 = hit_counter
        elif n_top == 1:
            top_1 = hit_counter

        label_bug_counter = 0
    return top_5, top_3, top_1


def results(log_path, data_name, codegen_size):
    total_top_5 = 0
    total_top_3 = 0
    total_top_1 = 0
    total_bugs = 0
    data_log_path = f"{log_path}/{data_name}"
    for subdir, _, files in os.walk(data_log_path):
        # print(file)
        if ".json" in file:
            f = open(os.path.join(subdir, file))

            subdir_name = subdir.replace(data_log_path + "/", "")

            split_dir = subdir_name.split("_")
            data_name = split_dir[0]
            params = split_dir[1]
            dimension = split_dir[2]

            if params == "256" or params == "512" or params == "1024":
                params = "scratch"

            if codegen_size != params:
                continue

            if "scratch" in dimension:
                dimension = ""
            data = json.load(f)
            probabilities = data["prob"]
            labels = data["label"]
            f.close()
            filtered_prob = []
            filtered_label = []
            for i, prob in enumerate(probabilities):
                if prob != 0:
                    filtered_prob.append(prob)
                    filtered_label.append(labels[i])
            label_name = f"{data_name}-{params}".replace("--", "-")

            top_5, top_3, top_1 = top_scores(probabilities, labels, label_name)
            total_top_5 += top_5
            total_top_3 += top_3
            total_top_1 += top_1
            if "bugsinpy" in label_name:
                total_bugs = 493
            elif "defects4j-1.2.0" in label_name:
                total_bugs = 226
                print(total_bugs)
            elif "defects4j" in label_name:
                total_bugs = 395
            elif "devign" in label_name:
                total_bugs = 5260

    if total_bugs:
        print(
            f"Top 1,3,5 of {total_bugs} total bugs for {data_name}-{codegen_size}: [{total_top_1}({top_ratio(total_top_1,total_bugs)}\%) & {total_top_3}({top_ratio(total_top_3,total_bugs)}\%) & {total_top_5}({top_ratio(total_top_5,total_bugs)}\%)]"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", help="Path to data root")
    ap.add_argument("pretrain_type", help="Pretrain size")

    args = ap.parse_args()
    log_path = args.log_path
    pretrain_type = args.pretrain_type
    data_name = "defects4j"

    current_path = os.getcwd()
    log_path = f"{current_path}/logs_path"

    results(log_path, data_name, pretrain_type)
