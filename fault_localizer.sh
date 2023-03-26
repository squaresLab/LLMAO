data_path=/code/data
data_name=bugsinpy # bugsinpy defects4j devign github/decoded
pretrain_type=350M # 350M, 2B, 6B, 16B
pretraining=0
python3 finetune_training.py $data_path $data_name $pretrain_type $pretraining