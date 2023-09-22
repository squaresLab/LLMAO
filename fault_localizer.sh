data_path=/home/data
data_name=defects4j-1.2.0 # bugsinpy defects4j devign github/decoded
pretrain_type=16B # 350M, 2B, 6B, 16B
pretraining=1
python3 training.py $data_path $data_name $pretrain_type $pretraining