<h1 align="center">
Nero optimiser
</h1>

## ImageNet training in PyTorch

This code is forked from [Pytorch examples](https://github.com/pytorch/examples/tree/master/imagenet).

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
- Install NVIDIA DALI

## Training
Assume you have 4 GPUs with >=11GB vRAM each.

Training with SGD
```bash
python main.py -a resnet50 --lr 0.1  DATASET_DIR --optimizer sgd --sch cos --workers 24 -b 400 --epochs 90 --momentum 0.9 --wd 0.0001 --epoch 90 
```

Training with SGD without weight decay
```bash
python main.py -a resnet50 --lr 0.1  DATASET_DIR --optimizer sgd --sch cos --workers 24 -b 400 --epochs 90 --momentum 0.9 --wd 0.0 --epoch 90 
```

Training with Nero without weight decay
```bash
python main.py -a resnet50 --lr 0.01  DATASET_DIR --optimizer nero --sch cos --workers 24 -b 400 --epochs 90 --momentum 0.0 --wd 0.0 --epoch 90 
```
