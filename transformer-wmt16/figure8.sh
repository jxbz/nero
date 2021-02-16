# { CUDA_VISIBLE_DEVICES=0 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim nero -lr 0.0001 ;
#   CUDA_VISIBLE_DEVICES=0 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim nero -lr 0.001 ;
#   CUDA_VISIBLE_DEVICES=0 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim nero -lr 0.01 ;
#   CUDA_VISIBLE_DEVICES=0 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim nero -lr 0.1 ; } &

# { CUDA_VISIBLE_DEVICES=1 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim sgd -lr 0.0001 ;
#   CUDA_VISIBLE_DEVICES=1 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim sgd -lr 0.001 ;
#   CUDA_VISIBLE_DEVICES=1 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim sgd -lr 0.01 ;
#   CUDA_VISIBLE_DEVICES=1 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim sgd -lr 0.1 ; } &

# { CUDA_VISIBLE_DEVICES=2 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim adam -lr 0.0001 ;
#   CUDA_VISIBLE_DEVICES=2 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim adam -lr 0.001 ;
#   CUDA_VISIBLE_DEVICES=2 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim adam -lr 0.01 ;
#   CUDA_VISIBLE_DEVICES=2 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim adam -lr 0.1 ; } &

# { CUDA_VISIBLE_DEVICES=3 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim lamb -lr 0.0001 ;
#   CUDA_VISIBLE_DEVICES=3 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim lamb -lr 0.001 ;
#   CUDA_VISIBLE_DEVICES=3 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim lamb -lr 0.01 ;
#   CUDA_VISIBLE_DEVICES=3 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim lamb -lr 0.1 ; }

{ CUDA_VISIBLE_DEVICES=0 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim nero -lr 0.001 -seed 1 ;
  CUDA_VISIBLE_DEVICES=0 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim nero -lr 0.001 -seed 2 ; } &

{ CUDA_VISIBLE_DEVICES=1 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim sgd -lr 0.0001 -seed 1 ;
  CUDA_VISIBLE_DEVICES=1 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim sgd -lr 0.0001 -seed 2 ; } &

{ CUDA_VISIBLE_DEVICES=2 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim adam -lr 0.0001 -seed 1 ;
  CUDA_VISIBLE_DEVICES=2 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim adam -lr 0.0001 -seed 2 ; } &

{ CUDA_VISIBLE_DEVICES=3 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim lamb -lr 0.01 -seed 1 ;
  CUDA_VISIBLE_DEVICES=3 python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -epoch 100 -optim lamb -lr 0.01 -seed 2 ; }
