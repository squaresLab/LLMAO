import os
import json

def find_topscores(fl_path):
    top_1 = 0
    top_3 = 0
    top_5 = 0
    for subdir, _, files in os.walk(fl_path):
        for file in files:
            file_path = os.path.join(subdir, file)
            if "sus.json" in file_path:
                with open(file_path.replace("sus", "metadata")) as json_file:
                    meta_json = json.load(json_file)
                with open(file_path) as json_file:
                    sus_json = json.load(json_file)
                real_bugs = meta_json["bug_line_number"]
                temp = {val: key for key, val in sus_json.items()}
                sus_json = {val: key for key, val in temp.items()}

                for i, (key, value) in enumerate(sus_json.items()):
                    rank = i + 1
                    if int(key) in real_bugs:
                        if rank == 1:
                            top_1 += 1
                        if rank <= 3:
                            top_3 += 1
                        if rank <= 5:
                            top_5 += 1
                        break
    print(f"top 5: {top_5}")
    print(f"top 3: {top_3}")
    print(f"top 1: {top_1}")

if __name__ == "__main__":
    print(f"Top score for llmao_window")
    current_path = os.getcwd()
    score_dir = "score_llmao_window"
    fl_path = f"{current_path}/{score_dir}"
    find_topscores(fl_path)
    print(f"Top score for Transfer")
    score_dir = "score_transferfl"
    fl_path = f"{current_path}/{score_dir}"
    find_topscores(fl_path)
