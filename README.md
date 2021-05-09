# Conformer: Local Features Coupling Global Representations for Visual Recognition

This repository is built upon [DeiT](https://github.com/facebookresearch/deit) and [timm](https://github.com/rwightman/pytorch-image-models)

# Usage

First, install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Training
To train Conformer-Ti and Conformer-S on ImageNet on a single node with 8 gpus for 300 epochs run:

Conformer-Ti
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
OUTPUT='./output/Conformer_tiny_patch16_batch_1024_lr1e-3_300epochs'

python -m torch.distributed.launch --master_port 50130 --nproc_per_node=8 --use_env main.py \
                                   --model Conformer_tiny_patch16 \
                                   --data-set IMNET \
                                   --batch-size 128 \
                                   --lr 0.001 \
                                   --num_workers 4 \
                                   --data-path /data/user/Dataset/ImageNet_ILSVRC2012/ \
                                   --output_dir ${OUTPUT} \
                                   --epochs 300
```

Conformer-S
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
OUTPUT='./output/Conformer_small_patch16_batch_1024_lr1e-3_300epochs'

python -m torch.distributed.launch --master_port 50130 --nproc_per_node=8 --use_env main.py \
                                   --model Conformer_tiny_patch16 \
                                   --data-set IMNET \
                                   --batch-size 128 \
                                   --lr 0.001 \
                                   --num_workers 4 \
                                   --data-path /data/user/Dataset/ImageNet_ILSVRC2012/ \
                                   --output_dir ${OUTPUT} \
                                   --epochs 300
```
