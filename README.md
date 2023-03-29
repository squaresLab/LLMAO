## LLMAO-Replication
```
docker pull huggingface/transformers-pytorch-gpu
```
```
docker run -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --mount type=bind,src="path-to-local-directory",dst=/home huggingface/transformers-pytorch-gpu:4.21.0
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


