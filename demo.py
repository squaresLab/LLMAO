import torch.utils.checkpoint
import torch
import argparse
from transformer import VoltronTransformerPretrained, TokenizeMask


def buglines_prediction(checkpoint_path, demo_type):
    pretrain_type = '350M'
    dim_model = 1024
    num_layer = 2
    target_dim = 512
    num_head = 8
    model = VoltronTransformerPretrained(
        num_layer=num_layer, dim_model=dim_model, num_head=num_head, target_dim=target_dim
    )
    model.load_state_dict(torch.load(
        f'{checkpoint_path}/{demo_type}_{pretrain_type}'), strict=False)
    model.eval()
    tokenize_mask = TokenizeMask(pretrain_type)
    if demo_type == 'defects4j':
        code_file_path = '/home/demo_code.java'
    else:
        code_file_path = '/home/demo_code.c'
    with open(code_file_path) as f:
        code_file = f.readlines()
        filtered_code = []
        for code_line in code_file:
            if demo_type =='defects4j' and code_line and '/' not in code_line and '*' not in code_line and code_line not in filtered_code:
                filtered_code.append(code_line)
            elif demo_type =='devign' and code_line and '/' not in code_line and '*' not in code_line and '#' not in code_line:
                filtered_code.append(code_line)

        code_lines = ''.join(filtered_code)
        input, mask, input_size, decoded_input = tokenize_mask.generate_token_mask(
            code_lines)
        input = input[None, :]
        mask = mask[None, :]
        predictions = model(input, mask)
        probabilities = torch.flatten(torch.sigmoid(predictions))
        real_indices = torch.flatten(mask == 1)            
        probabilities = probabilities[real_indices].tolist()        
        decoded_input_list = decoded_input.split('\n')
        decoded_input = [line.lstrip('\t')
                            for line in decoded_input_list]
        decoded_input = "\n".join(decoded_input)
        probabilities = probabilities[:input_size+1]
        most_sus = list(
            map(lambda x: 1 if x > 0.15 else 0, probabilities))
        result_dict = []
        for i, p in enumerate(most_sus):
            if p == 1 and len(filtered_code[i].strip()) > 1:
                result_dict.append({"line": i, "score": round(probabilities[i]*100,2)})

        result_dict = sorted(result_dict, key=lambda d: d['score'], reverse=True)
        for res in result_dict:
            if demo_type == 'defects4j':
                bug_index = res["line"]-1 # Java codegen tokenizer offset
            else:
                bug_index = res["line"]
            print(f'line-{res["line"]} sus-{res["score"]}%: {filtered_code[bug_index]}')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_path")
    ap.add_argument("demo_type")
    args = ap.parse_args()
    checkpoint_path = args.checkpoint_path
    demo_type = args.demo_type
    buglines_prediction(checkpoint_path, demo_type)
