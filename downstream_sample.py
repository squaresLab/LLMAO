import torch.utils.checkpoint
import torch
from glob import glob
import pandas as pd
import os
import argparse
from transformer import VoltronTransformerPretrained, TokenizeMask


def buglines_prediction(data_path):
    buggy_lines_list = []
    pretrain_type = '2B'
    if pretrain_type == '350M':
        dim_model = 1024
    elif pretrain_type == '2B':
        dim_model = 2560
    elif pretrain_type == '6B':
        dim_model = 4096
    elif pretrain_type == '16B':
        dim_model = 6144
    num_layer = 2
    target_dim = 1024
    if target_dim == 1024:
        num_head = 16
    elif target_dim == 512:
        num_head = 8
    elif target_dim == 256:
        num_head = 4
    model = VoltronTransformerPretrained(
        num_layer=num_layer, dim_model=dim_model, num_head=num_head, target_dim=target_dim
    )
    tokenize_mask = TokenizeMask()
    model.load_state_dict(torch.load(
        f'{data_path}/model_checkpoints/defects4j_{pretrain_type}'))
    model.eval()

    with open('/code/torch_transformer/test_input.txt') as f:
        code = f.readlines()
        code_lines = ''.join(code)
        input, mask, input_size, decoded_input = tokenize_mask.generate_token_mask(
            code_lines)
        input = input[None, :]
        mask = mask[None, :]
        predictions = model(input, mask)
        # print(predictions.shape)
        probabilities = torch.flatten(
            torch.sigmoid(predictions)).tolist()
        # print(len(probabilities))
        decoded_input_list = decoded_input.split('\n')
        decoded_input = [line.lstrip('\t')
                            for line in decoded_input_list]
        decoded_input = "\n".join(decoded_input)
        probabilities = probabilities[:input_size+1]
        probabilities = list(
            map(lambda x: 1 if x > 0.2 else 0, probabilities))
        
        for i, p in enumerate(probabilities):
            if p == 1:
                print(i)

        buggy_lines = []
        for i, p in enumerate(probabilities):
            if p > 0.2:
                buggy_lines.append(i)
        buggy_lines_list.append(
            {'code': decoded_input, 'fault_probability': probabilities})

    code_df = pd.DataFrame.from_records(buggy_lines_list)
    code_df.to_csv(f'{data_path}/downstream_output/test_sample.csv',
                   mode="w+", header=False, index=False)



if __name__ == "__main__":
    root_path = '/code/data'
    buglines_prediction(root_path)
