# LLMAO-Replication

I. Requirements
--------------------
We recommend using docker for LLMAO.

Pull the huggingface docker image, which includes most requirements

`docker pull huggingface/transformers-pytorch-gpu`

Run a container. Make sure to mount the container to your own directory path. We assume a GPU exists, as training a LLM requires a significant amount of GPU VRAM. If you do not have a GPU, simply remove the line 
`--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0` below.

`docker run -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --mount type=bind,src="path-to-local-directory",dst=/home huggingface/transformers-pytorch-gpu:4.21.0`



docker run --name=dan236 -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=2,3,6 \
--mount type=bind,src=/home/aidan/,dst=/home \
--mount type=bind,src=/data/huggingface/,dst=/models/huggingface \
--mount type=bind,src=/data/aidan/voltron_data,dst=/home/data aidan
XDG_CACHE_HOME='/models'
export XDG_CACHE_HOME
echo $XDG_CACHE_HOME
conda activate torch


Install some additional dependencies
```
pip install --upgrade pip
pip install accelerate
pip install torchdata
```

II. Obtain results quickly
---------------------------
Top scores:

`bash results_topscores.sh`

ROC plots and AUC scores:

`bash results_plot.sh`

III. Demo
---------------------------
`bash demo.sh`
Change `demo_type='devign'` in demo.sh for a demo of security vulnerability detection.


IV. Train model yourself
---------------------------
Download Dataset
1. Click the following url link and download the dataset used in this research.

    [data](https://mega.nz/folder/hHIjjZoA#v2BxPdzMlHwH0gBDg9oUjQ)

2. Unzip it and put the folder in the same path as this repo

3. Load Codegen final hidden states:
    change `biggest_model=1` to use Codegen-16B: requires significant amount of GPU vram and storage.

    `bash codegen_loading.sh`

4. Train 

    `bash fault_localizer.sh`

5. Reload results

    Change log_path=/home/model_logs_tenfold in results_topscores.sh to log_path=/home/model_logs

    `python3 top_scores.py`

    `python3 plotter.py`

```


