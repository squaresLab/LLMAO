import os
import json
from itertools import chain
import torch.utils.checkpoint
import torch
from transformer import VoltronTransformerPretrained, TokenizeMask


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


def sort_by_value_desc(dic):
    return {
        k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)
    }


def load_model(pretrain_type):
    num_layer = 2
    target_dim = 512
    num_head = 8
    if pretrain_type == "16B":
        target_dim = 1024
        dim_model = 6144
        num_head = 16
    elif pretrain_type == "6B":
        dim_model = 4096
    elif pretrain_type == "350M":
        dim_model = 1024
    model = VoltronTransformerPretrained(
        num_layer=num_layer,
        dim_model=dim_model,
        num_head=num_head,
        target_dim=target_dim,
    )
    model.load_state_dict(
        torch.load(f"llmao/model_checkpoints/defects4j_{pretrain_type}"), strict=False
    )
    model.eval()

    return model


def llmao_prediction(model, tokenize_mask, filtered_code):
    code_lines = "".join(filtered_code)
    try:
        input, mask, input_size, decoded_input = tokenize_mask.generate_token_mask(
            code_lines
        )
    except:
        return {}
    input = input[None, :]
    mask = mask[None, :]
    predictions = model(input, mask)
    probabilities = torch.flatten(torch.sigmoid(predictions))
    real_indices = torch.flatten(mask == 1)
    probabilities = probabilities[real_indices].tolist()
    decoded_input_list = decoded_input.split("\n")
    decoded_input = [line.lstrip("\t") for line in decoded_input_list]
    decoded_input = "\n".join(decoded_input)
    probabilities = probabilities[: input_size + 1]
    most_sus = list(map(lambda x: 1 if x > 0 else 0, probabilities))
    result_list = []
    for i, _ in enumerate(most_sus):
        result_list.append(
            {"code": filtered_code[i - 1], "score": round(probabilities[i], 5)}
        )

    ## sort and filter top 10 sus_scores
    result_list = sorted(result_list, key=lambda d: d["score"], reverse=True)
    #########
    return result_list


def use_window_method(model, tokenize_mask, code_lines):
    result_dict = {}
    # Process code lines and divide into chunks of 124 lines
    filtered_code_lines = []
    for code_line in code_lines:
        if (
            code_line
            and not code_line.strip().startswith("/")
            and not code_line.strip().startswith("*")
            and not code_line.strip().startswith("#")
            and len(code_line.strip()) > 0
        ):
            filtered_code_lines.append(code_line)
    code_broken = list(divide_chunks(filtered_code_lines, 124))
    # Run LLMAO on each chunk
    entire_file_predictions = []
    for code_chunk in code_broken:
        llmao_list = llmao_prediction(model, tokenize_mask, code_chunk)
        llmao_list = llmao_list[:15]
        entire_file_predictions.append(llmao_list)
    result_list = chain.from_iterable(entire_file_predictions)
    result_list = sorted(result_list, key=lambda d: d["score"], reverse=True)
    result_list = result_list[:124]
    # LLMAO on resulting combined list
    result_list = llmao_prediction(
        model, tokenize_mask, [res["code"] for res in result_list]
    )
    result_list = result_list[:50]
    for res in result_list:
        llmao_code = res["code"]
        for i in range(len(code_lines)):
            if code_lines[i] == llmao_code:
                result_dict[i + 1] = res["score"]
                break
        # result_dict[res["line"]] = res["score"]
    return result_dict


def llmao_gen(pretrain_type, output_dir):
    model = load_model(pretrain_type)
    tokenize_mask = TokenizeMask(pretrain_type)
    current_path = os.getcwd()
    priorfl_path = f"{current_path}/score_transferfl"
    for subdir, _, files in os.walk(priorfl_path):
        for file in files:
            file_path = os.path.join(subdir, file)
            bug_num = subdir.split("/")[-1]
            d4j_proj = subdir.split("/")[-2]
            if not os.path.exists(f"{output_dir}{d4j_proj}"):
                os.mkdir(f"{output_dir}{d4j_proj}")
            if not os.path.exists(f"{output_dir}{d4j_proj}/{bug_num}"):
                os.mkdir(f"{output_dir}{d4j_proj}/{bug_num}")
            else:
                continue
            sbfl_list = []
            code_lines = []
            if "sus.json" in file_path:
                with open(file_path.replace("sus", "metadata")) as json_file:
                    meta_json = json.load(json_file)
                    code_path = f"d4j_code/{d4j_proj}/{bug_num}/b{bug_num}.java"
                    with open(code_path, "r") as jcode:
                        code = jcode.readlines()
                with open(file_path) as json_file:
                    sus_json = json.load(json_file)
                sus_json = {
                    k: v
                    for k, v in sorted(
                        sus_json.items(), key=lambda item: item[1], reverse=True
                    )
                }
                counter = 0
                for line_num, prediction in sus_json.items():
                    if int(line_num) > len(code):
                        continue
                    sus_line = code[int(line_num) - 1]
                    sbfl_list.append(
                        {"line": line_num, "code": sus_line, "score": prediction}
                    )
                    code_lines.append(sus_line)
                    counter += 1
                    if counter > 10:
                        break

                result_dict = use_window_method(model, tokenize_mask, code)

                # using filter
                test_dict = {k: v for k, v in result_dict.items() if str(k) in sus_json}
                if test_dict:
                    result_dict = test_dict
                print(
                    f"Writing {d4j_proj} {bug_num} to {output_dir}{d4j_proj}/{bug_num}/sus.json"
                )
                write_file = f"{output_dir}{d4j_proj}/{bug_num}/sus.json"
                meta_file = f"{output_dir}{d4j_proj}/{bug_num}/metadata.json"
                with open(os.path.join(subdir, write_file), "w") as fp:
                    json.dump(result_dict, fp)
                with open(os.path.join(subdir, meta_file), "w") as fp:
                    json.dump(meta_json, fp)


if __name__ == "__main__":
    output_dir = "score_llmao_window/"
    llmao_gen("16B", output_dir)
