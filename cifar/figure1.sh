# command to reproduce ablation experiments
# Fine grained LR search in the range of 0.01 to 0.1
# No constraints: 
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.02
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.04
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.06
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.08
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.10

# Mean constraint: 
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01 --c1
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.02 --c1
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.04 --c1
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.06 --c1
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.08 --c1
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.10 --c1

# Norm constraints: 
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01 --c2
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.02 --c2
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.04 --c2
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.06 --c2
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.08 --c2
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.10 --c2

# Both constraints: 
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01 --c1 --c2
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.02 --c1 --c2
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.04 --c1 --c2
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.06 --c1 --c2
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.08 --c1 --c2
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.10 --c1 --c2

#Best lr repeats
# No constraints:
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 1  --lr 0.02
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 2  --lr 0.02

# Mean constraint:
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 1  --lr 0.01 --c1
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 2  --lr 0.01 --c1

# Norm constraint:
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 1  --lr 0.02 --c2
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 2  --lr 0.02 --c2

# Both constraints:
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 1  --lr 0.02 --c1 --c2
env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 2  --lr 0.02 --c1 --c2