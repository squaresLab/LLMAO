## LLMAO-Replication

```
docker pull huggingface/transformers-pytorch-gpu
```

```
docker run --name=dan3 -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=3 --mount type=bind,src=/home/aidan/LLMAO,dst=/home huggingface/transformers-pytorch-gpu:4.21.0
```

```
docker run --name=dan9 -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=9 --mount type=bind,src=/home/aidan/LLMAO,dst=/home --mount type=bind,src=/data/huggingface/,dst=/models/huggingface aidan
XDG_CACHE_HOME='/models'
export XDG_CACHE_HOME
```

```
pip install --upgrade pip
pip install accelerate
pip install torchdata
```


# To obtain results quickly
Top scores:
bash results_topscores.sh

ROC plots and AUC scores:
bash results_plot.sh

# To train model yourself

place /data in LLMAO main repository
Load codegen final hidden states
```
bash codegen_loading.sh
```

# Train model
```
bash fault_localizer.sh
```

# Results
```
Change log_path=/home/model_logs_tenfold in results_topscores.sh to log_path=/home/model_logs
python3 top_scores.py
python3 plotter.py

```


