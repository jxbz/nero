<h1 align="center">
Nero optimiser
</h1>

## Attention is all you need: A Pytorch Implementation

This code is forked from [jadore801120's implementation](https://github.com/jadore801120/attention-is-all-you-need-pytorch). We made the following modifications to the repo:
- simplified the LR schedule to linear warmup and warmdown.
- removed logit scaling from the decoder output---this change was suggested in [this pull request](https://github.com/jadore801120/attention-is-all-you-need-pytorch/pull/168) to improve the baseline performance.
- set elementwise affine to False in layer norm. This was done at a very preliminary stage when we were still unsure how we wanted to implement bias updates in Nero. We never re-activated it.

## Requirements
- python 3.4+
- pytorch 1.3.1
- torchtext 0.4.0
- spacy 2.2.2+
- tqdm
- dill
- numpy

## Usage

### 0) Download the spacy language model.
```bash
# conda install -c conda-forge spacy 
python -m spacy download en
python -m spacy download de
```

### 1) Preprocess the data with torchtext and spacy.
```bash
python preprocess.py -lang_src de -lang_trg en -share_vocab -save_data m30k_deen_shr.pkl
```

### 2) Train the model
See the training commands in `figure8.sh`.
