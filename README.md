# LLMAO-Replication
docker pull huggingface/transformers-pytorch-gpu

docker run --name=dan3 -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=3 --mount type=bind,src=/home/aidan/LLMAO,dst=/home huggingface/transformers-pytorch-gpu:4.21.0



