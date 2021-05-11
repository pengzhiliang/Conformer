# Conformer: Local Features Coupling Global Representations for Visual Recognition ([arxiv](https://arxiv.org/abs/2105.03889))

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
To train Conformer-S on ImageNet on a single node with 8 gpus for 300 epochs run:

Conformer-S
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
OUTPUT='./output/Conformer_small_patch16_batch_1024_lr1e-3_300epochs'

python -m torch.distributed.launch --master_port 50130 --nproc_per_node=8 --use_env main.py \
                                   --model Conformer_small_patch16 \
                                   --data-set IMNET \
                                   --batch-size 128 \
                                   --lr 0.001 \
                                   --num_workers 4 \
                                   --data-path /data/user/Dataset/ImageNet_ILSVRC2012/ \
                                   --output_dir ${OUTPUT} \
                                   --epochs 300
```

## Model Zoo

| Model        | Parameters | MACs   | Top-1 Acc | Link |
| ------------ | ---------- | ------ | --------- | ---- |
| Conformer-Ti | 23.5 M     | 5.2 G  | 81.3 %    | [baidu](https://pan.baidu.com/s/12AblBmhUu5gnYsPjnDE_Jg)(code: hzhm) [google]() |
| Conformer-S  | 37.7 M     | 10.6 G | 83.4 %    | [baidu](https://pan.baidu.com/s/1kYOZ9mRP5fvujH6snsOjew)(code: qvu8) [google]() |
| Conformer-B  | 83.3 M     | 23.3 G | 84.1 %    | [baidu](https://pan.baidu.com/s/1FL5XDAqHoimpUxNSunKq0w)(code: b4z9) [google]() |

## Citation
```
@article{peng2021conformer,
      title={Conformer: Local Features Coupling Global Representations for Visual Recognition}, 
      author={Zhiliang Peng and Wei Huang and Shanzhi Gu and Lingxi Xie and Yaowei Wang and Jianbin Jiao and Qixiang Ye},
      journal={arXiv preprint arXiv:2105.03889},
      year={2021},
}
```
