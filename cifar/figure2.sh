# command to reproduce Nero vs Madam vs LAMB experiments
# Fine grained LR search in the range of 0.01 to 0.1
# No constraints: 
# Nero
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.02
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.04
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.06
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.08
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.10

# LAMB: 
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.02
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.04
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.06
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.08
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.10

# Madam: 
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer madam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer madam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.02
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer madam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.04
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer madam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.06
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer madam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.08
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer madam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.10

#Best lr repeats:
# Nero:
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 1  --lr 0.02
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer neroabl --momentum 0.0 --beta 0.999 --wd 0.0 --seed 2  --lr 0.02

# LAMB:
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 1  --lr 0.02
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer lamb --momentum 0.0 --beta 0.999 --wd 0.0 --seed 2  --lr 0.02

# Madam:
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer madam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 1  --lr 0.02
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer madam --momentum 0.0 --beta 0.999 --wd 0.0 --seed 2  --lr 0.02


# With constraints:
# Nero 
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.02
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.04
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.06
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.08
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.10

# LAMB: 
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer lambcs --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer lambcs --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.02
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer lambcs --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.04
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer lambcs --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.06
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer lambcs --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.08
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer lambcs --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.10

# Madam: 
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer madamcs --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer madamcs --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.02
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer madamcs --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.04
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer madamcs --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.06
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer madamcs --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.08
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer madamcs --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.10

#Best lr repeats:
# Nero:
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 1  --lr 0.02
env CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 2  --lr 0.02

# LAMB:
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer lambcs --momentum 0.0 --beta 0.999 --wd 0.0 --seed 1  --lr 0.06
env CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer lambcs --momentum 0.0 --beta 0.999 --wd 0.0 --seed 2  --lr 0.06

# Madam:
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer madamcs --momentum 0.0 --beta 0.999 --wd 0.0 --seed 1  --lr 0.06
env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer madamcs --momentum 0.0 --beta 0.999 --wd 0.0 --seed 2  --lr 0.06
