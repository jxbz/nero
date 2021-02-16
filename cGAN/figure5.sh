# LR search:
# Nero
CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --optim nero --initial_lr 0.0001
CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --optim nero --initial_lr 0.001
CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --optim nero --initial_lr 0.01
CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --optim nero --initial_lr 0.1
CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --optim nero --initial_lr 1.0

# SGD
CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --optim sgd --initial_lr 0.0001
CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --optim sgd --initial_lr 0.001
CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --optim sgd --initial_lr 0.01
CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --optim sgd --initial_lr 0.1
CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --optim sgd --initial_lr 1.0

# Adam
CUDA_VISIBLE_DEVICES=2 python main.py --seed 0 --optim adam --initial_lr 0.0001
CUDA_VISIBLE_DEVICES=2 python main.py --seed 0 --optim adam --initial_lr 0.001
CUDA_VISIBLE_DEVICES=2 python main.py --seed 0 --optim adam --initial_lr 0.01
CUDA_VISIBLE_DEVICES=2 python main.py --seed 0 --optim adam --initial_lr 0.1
CUDA_VISIBLE_DEVICES=2 python main.py --seed 0 --optim adam --initial_lr 1.0

# LAMB
CUDA_VISIBLE_DEVICES=3 python main.py --seed 0 --optim lamb --initial_lr 0.0001
CUDA_VISIBLE_DEVICES=3 python main.py --seed 0 --optim lamb --initial_lr 0.001
CUDA_VISIBLE_DEVICES=3 python main.py --seed 0 --optim lamb --initial_lr 0.01
CUDA_VISIBLE_DEVICES=3 python main.py --seed 0 --optim lamb --initial_lr 0.1
CUDA_VISIBLE_DEVICES=3 python main.py --seed 0 --optim lamb --initial_lr 1.0

# Best LR repeats:
# Nero
CUDA_VISIBLE_DEVICES=0 python main.py --seed 1 --optim nero --initial_lr 0.01
CUDA_VISIBLE_DEVICES=0 python main.py --seed 2 --optim nero --initial_lr 0.01

# SGD
CUDA_VISIBLE_DEVICES=1 python main.py --seed 1 --optim sgd --initial_lr 0.01
CUDA_VISIBLE_DEVICES=1 python main.py --seed 2 --optim sgd --initial_lr 0.01

# Adam
CUDA_VISIBLE_DEVICES=2 python main.py --seed 1 --optim adam --initial_lr 0.0001
CUDA_VISIBLE_DEVICES=2 python main.py --seed 2 --optim adam --initial_lr 0.0001

# LAMB
CUDA_VISIBLE_DEVICES=3 python main.py --seed 1 --optim lamb --initial_lr 0.01
CUDA_VISIBLE_DEVICES=3 python main.py --seed 2 --optim lamb --initial_lr 0.01