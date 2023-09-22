# LLMAO-Replication

I. Requirements
--------------------
We recommend using docker for LLMAO.
```
# Pull the huggingface docker image, which includes most requirements

docker pull huggingface/transformers-pytorch-gpu

# Run a container. Make sure to mount the container to your own directory path. We assume an Nvidia GPU exists, as training and loading an LLM requires a significant amount of GPU VRAM.

docker run -it --mount type=bind,src="path-to-local-directory",dst=/home huggingface/transformers-pytorch-gpu:4.21.0

# Install some additional dependencies
pip install --upgrade pip
pip install accelerate
pip install torchdata
```

II. Obtain results quickly
---------------------------
Top scores:
```
python3 top_scores.py model_logs $pretrain_type
# Example
python3 top_scores.py model_logs 16B
```



ROC plots and AUC scores:
```
python3 plotter.py plotfiles
```

III. Demo
---------------------------
We include two example code files here for demonstration: `demo_code.c` and `demo_code.java`.

With actual buggy lines 93, 95 for `demo_code.c`,
and actual buggy lines 20, 25 for `demo_code.java`.

```
python3 demo.py $demo_type $pretrain_type $code_file_path
example: python3 demo.py devign 350M demo_code.c


output: 
line-95 sus-21.35%:         tcg_gen_ext16u_i32(QREG_DIV1, reg);
line-93 sus-17.59%:         tcg_gen_ext16s_i32(QREG_DIV1, reg);
```

Minimum VRAM (GPU memory) required for loading each of the checkpoints:
350M: 2.6GB
6B: 14.7GB
16B: 38GB (recommend at least 2-3 GPUs)

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




