<h1 align="center">
Nero optimiser
</h1>

# Pytorch-cifar10

This code is forked from [Pytorch Cifar](https://github.com/weiaicunzai/pytorch-cifar100).

## Requirements

Tested with:
- python3.7
- pytorch1.5.1+cu101
- tensorboard (optional)

## Usage

### 1. enter directory
```bash
$ cd cifar
```

### 2. train the model
You need to specify the net you want to train using arg -net

```bash
# use gpu to train vgg11
$ python train.py -net vgg11 -gpu
```

### 3. test the model
Test the model using test.py
```bash
$ python test.py -net vgg16 -weights path_to_vgg16_weights_file
```

## Implementated NetWork

- vgg [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556v6)

- resnet [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385v1)




