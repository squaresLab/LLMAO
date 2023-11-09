# LLMAO

LLMAO is a large language model based fault localization tool, associated with the following [paper](https://arxiv.org/abs/2310.01726) published at ICSE-2024.

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


ROC plots and AUC scores:
```
python3 plotter.py plotfiles
```

II. Demo
---------------------------
We include two example code files here for demonstration: `demo_code.c` and `demo_code.java`.

With actual vulnerable lines 52-62 for `demo_code.c`,
and actual buggy lines 20-30 for `demo_code.java`.

```
python3 demo.py $demo_type $pretrain_type $code_file_path
example: python3 demo.py devign 16B demo_code.c


output: 
line-52 sus-15.86%:         DISAS_INSN(divw)
...
```

Minimum VRAM (GPU memory) required for loading each of the checkpoints:

350M: 2.6GB

6B: 14.7GB

16B: 38GB (recommend at least 2-3 GPUs)


III. Obtain some top scores
---------------------------
Top scores:
```
python3 top_scores.py model_logs $pretrain_type
# Example
python3 top_scores.py model_logs 16B
```


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

5. Rerun results
    `python3 top_scores.py`
    `python3 plotter.py`


V. Run LLMAO on file level
---------------------------
LLMAO was neither trained nor evaluated on file level due to its limited context window of 128 lines.
The training and evaluation procedure of LLMAO is described in Section 3.1 of the paper.

To run LLMAO on a much larger file, one way is to split the file into multiple chunks of 128 lines and combine scores at the end. 
However, this way of using LLMAO removes valuable context across the entire file, and buggy or vulnerable lines across multiple chunks cannot be accuracy detected.
We include the method for running LLMAO on Defects4J entire files in this replication package to showcase the limitiation of our LLM-based fault localization.
We hope that this limitation can be reduced as LLMs grow larger and can process significantly larger context windows.
Enter the following:
```
python3 top_score_window.py
```
Output:

```
Top score for llmao_window
top 5: 77
top 3: 52
top 1: 24

Top score for Transfer
top 5: 145
top 3: 126
top 1: 69
```

In which LLMAO has much weaker results than Transfer-FL, a prior fault localization approach that is trained on Defects4J for each individual bug.

To remake LLMAO file level scores:
```
python3 llmao_d4j_window.py
```


