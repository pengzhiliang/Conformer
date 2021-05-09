#!/usr/bin/env bash

# Train
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
OUTPUT='./output/Conformer_small_patch16_batch_1024_lr1e-3_300epochs'

python -m torch.distributed.launch --master_port 50132 --nproc_per_node=8 --use_env main.py \
                                   --model Conformer_small_patch16 \
                                   --data-set IMNET \
                                   --batch-size 128 \
                                   --lr 0.001 \
                                   --num_workers 4 \
                                   --data-path /data/user/Dataset/ImageNet_ILSVRC2012/ \
                                   --output_dir ${OUTPUT} \
                                   --epochs 300

# Inference
#CUDA_VISIBLE_DEVICES=0, python main.py  --model Conformer_tiny_patch16 --eval --batch-size 64 \
#                --input-size 224 \
#                --data-set IMNET \
#                --num_workers 4 \
#                --data-path ../ImageNet_ILSVRC2012/ \
#                --epochs 100 \
#                --resume ../Conformer_tiny_patch16.pth


