# command to reproduce wikitext-2 experiments
# LR search
# Nero
env CUDA_VISIBLE_DEVICES=0 python main.py --cuda --epochs 20 --model Transformer --optim nero  --lr 0.0001 --seed 0
env CUDA_VISIBLE_DEVICES=0 python main.py --cuda --epochs 20 --model Transformer --optim nero  --lr 0.001 --seed 0
env CUDA_VISIBLE_DEVICES=0 python main.py --cuda --epochs 20 --model Transformer --optim nero  --lr 0.01 --seed 0
env CUDA_VISIBLE_DEVICES=0 python main.py --cuda --epochs 20 --model Transformer --optim nero  --lr 0.1 --seed 0
env CUDA_VISIBLE_DEVICES=0 python main.py --cuda --epochs 20 --model Transformer --optim nero  --lr 1.0 --seed 0

# SGD 
env CUDA_VISIBLE_DEVICES=1 python main.py --cuda --epochs 20 --model Transformer --optim sgd  --lr 0.0001 --seed 0
env CUDA_VISIBLE_DEVICES=1 python main.py --cuda --epochs 20 --model Transformer --optim sgd  --lr 0.001 --seed 0
env CUDA_VISIBLE_DEVICES=1 python main.py --cuda --epochs 20 --model Transformer --optim sgd  --lr 0.01 --seed 0
env CUDA_VISIBLE_DEVICES=1 python main.py --cuda --epochs 20 --model Transformer --optim sgd  --lr 0.1 --seed 0
env CUDA_VISIBLE_DEVICES=1 python main.py --cuda --epochs 20 --model Transformer --optim sgd  --lr 1.0 --seed 0

# Adam 
env CUDA_VISIBLE_DEVICES=2 python main.py --cuda --epochs 20 --model Transformer --optim adam  --lr 0.0001 --seed 0
env CUDA_VISIBLE_DEVICES=2 python main.py --cuda --epochs 20 --model Transformer --optim adam  --lr 0.001 --seed 0
env CUDA_VISIBLE_DEVICES=2 python main.py --cuda --epochs 20 --model Transformer --optim adam  --lr 0.01 --seed 0
env CUDA_VISIBLE_DEVICES=2 python main.py --cuda --epochs 20 --model Transformer --optim adam  --lr 0.1 --seed 0
env CUDA_VISIBLE_DEVICES=2 python main.py --cuda --epochs 20 --model Transformer --optim adam  --lr 1.0 --seed 0

# LAMB
env CUDA_VISIBLE_DEVICES=3 python main.py --cuda --epochs 20 --model Transformer --optim lamb  --lr 0.0001 --seed 0
env CUDA_VISIBLE_DEVICES=3 python main.py --cuda --epochs 20 --model Transformer --optim lamb  --lr 0.001 --seed 0
env CUDA_VISIBLE_DEVICES=3 python main.py --cuda --epochs 20 --model Transformer --optim lamb  --lr 0.01 --seed 0
env CUDA_VISIBLE_DEVICES=3 python main.py --cuda --epochs 20 --model Transformer --optim lamb  --lr 0.1 --seed 0
env CUDA_VISIBLE_DEVICES=3 python main.py --cuda --epochs 20 --model Transformer --optim lamb  --lr 1.0 --seed 0

#Best lr repeats:
# Nero
env CUDA_VISIBLE_DEVICES=0 python main.py --cuda --epochs 20 --model Transformer --optim nero  --lr 0.01 --seed 1
env CUDA_VISIBLE_DEVICES=0 python main.py --cuda --epochs 20 --model Transformer --optim nero  --lr 0.01 --seed 2

# SGD
env CUDA_VISIBLE_DEVICES=1 python main.py --cuda --epochs 20 --model Transformer --optim sgd  --lr 1.0 --seed 1
env CUDA_VISIBLE_DEVICES=1 python main.py --cuda --epochs 20 --model Transformer --optim sgd  --lr 1.0 --seed 2

# Adam
env CUDA_VISIBLE_DEVICES=2 python main.py --cuda --epochs 20 --model Transformer --optim adam  --lr 0.0001 --seed 1
env CUDA_VISIBLE_DEVICES=2 python main.py --cuda --epochs 20 --model Transformer --optim adam  --lr 0.0001 --seed 2

# LAMB
env CUDA_VISIBLE_DEVICES=3 python main.py --cuda --epochs 20 --model Transformer --optim lamb  --lr 0.01 --seed 1
env CUDA_VISIBLE_DEVICES=3 python main.py --cuda --epochs 20 --model Transformer --optim lamb  --lr 0.01 --seed 2