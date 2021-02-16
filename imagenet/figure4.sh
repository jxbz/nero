# replace DATASET_DIR with your ImageNet dataset folder

# SGD with momentum and weight decay:
python main.py -a resnet50 --lr 0.1  DATASET_DIR --optimizer sgd --sch cos --workers 24 -b 400 --epochs 90 --momentum 0.9 --wd 0.0001 --epoch 90
python main.py -a resnet50 --lr 0.1  DATASET_DIR --optimizer sgd --sch cos --workers 24 -b 400 --epochs 90 --momentum 0.9 --wd 0.0001 --epoch 90
python main.py -a resnet50 --lr 0.1  DATASET_DIR --optimizer sgd --sch cos --workers 24 -b 400 --epochs 90 --momentum 0.9 --wd 0.0001 --epoch 90

# SGD with momentum, without weight decay:
python main.py -a resnet50 --lr 0.1  DATASET_DIR --optimizer sgd --sch cos --workers 24 -b 400 --epochs 90 --momentum 0.9 --wd 0.0 --epoch 90
python main.py -a resnet50 --lr 0.1  DATASET_DIR --optimizer sgd --sch cos --workers 24 -b 400 --epochs 90 --momentum 0.9 --wd 0.0 --epoch 90
python main.py -a resnet50 --lr 0.1  DATASET_DIR --optimizer sgd --sch cos --workers 24 -b 400 --epochs 90 --momentum 0.9 --wd 0.0 --epoch 90

# Nero out-of-the-box:
python main.py -a resnet50 --lr 0.1  DATASET_DIR --optimizer nero --sch cos --workers 24 -b 400 --epochs 90 --momentum 0.0 --wd 0.0 --epoch 90
python main.py -a resnet50 --lr 0.1  DATASET_DIR --optimizer nero --sch cos --workers 24 -b 400 --epochs 90 --momentum 0.0 --wd 0.0 --epoch 90
python main.py -a resnet50 --lr 0.1  DATASET_DIR --optimizer nero --sch cos --workers 24 -b 400 --epochs 90 --momentum 0.0 --wd 0.0 --epoch 90