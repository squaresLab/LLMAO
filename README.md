# To obtain results quickly
python3 top_scores.py
python3 plotter.py

# LLMAO-Replication
docker pull huggingface/transformers-pytorch-gpu
docker run --name=dan3 -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=3 --mount type=bind,src=/home/aidan/LLMAO,dst=/home huggingface/transformers-pytorch-gpu:4.21.0

pip install --upgrade pip
pip install accelerate
pip install torchdata


# place /data in LLMAO main repository

# Load codegen final hidden states
bash codegen_loading.sh

# Train model
bash fault_localizer.sh

# Results


chmod -R a+rw model_logs


