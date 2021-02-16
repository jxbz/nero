# command to reproduce Nero vs SGD vs Adam vs LAMB experiments
# VGG-11
# LR search
# Nero
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.0001
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.001
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.1
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 1.0

# SGD 
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer sgd --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.0001
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer sgd --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.001
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer sgd --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer sgd --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.1
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer sgd --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 1.0

# Adam 
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer adam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.0001
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer adam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.001
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer adam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer adam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.1
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer adam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 1.0

# LAMB
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.0001
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.001
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.1
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 1.0

#Best lr repeats:
# Nero
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 1  --lr 0.01
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 2  --lr 0.01

# SGD
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer sgd --momentum 0.0 --beta 0.999 --wd 0.0 --seed 1  --lr 0.1
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer sgd --momentum 0.0 --beta 0.999 --wd 0.0 --seed 2  --lr 0.1

# Adam
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer adam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 1  --lr 0.001
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer adam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 2  --lr 0.001

# LAMB
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 1  --lr 0.01
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 2  --lr 0.01


# ResNet-18
# LR search
# Nero 
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net resnet18 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.0001
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net resnet18 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.001
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net resnet18 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net resnet18 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.1
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net resnet18 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 1.0

# SGD
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net resnet18 --optimizer sgd --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.0001
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net resnet18 --optimizer sgd --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.001
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net resnet18 --optimizer sgd --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net resnet18 --optimizer sgd --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.1
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net resnet18 --optimizer sgd --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 1.0

# Adam
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net resnet18 --optimizer adam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.0001
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net resnet18 --optimizer adam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.001
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net resnet18 --optimizer adam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net resnet18 --optimizer adam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.1
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net resnet18 --optimizer adam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 1.0

# LAMB
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net resnet18 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.0001
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net resnet18 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.001
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net resnet18 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net resnet18 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.1
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net resnet18 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 1.0


#Best lr repeats:
# Nero
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net resnet18 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 1  --lr 0.01
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net resnet18 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 2  --lr 0.01

# SGD
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net resnet18 --optimizer sgd --momentum 0.0 --beta 0.999 --wd 0.0 --seed 1  --lr 0.1
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net resnet18 --optimizer sgd --momentum 0.0 --beta 0.999 --wd 0.0 --seed 2  --lr 0.1

# Adam
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net resnet18 --optimizer adam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 1  --lr 0.01
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net resnet18 --optimizer adam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 2  --lr 0.01

# LAMB
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net resnet18 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 1  --lr 0.1
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net resnet18 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 2  --lr 0.1
