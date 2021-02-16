<h1 align="center">
Nero optimiser
</h1>

<p align="center">
  <img src="soccer.svg" width="150"/>
</p>

<p align="center">
  <a href="https://scholar.google.com/citations?user=nVWQwHkAAAAJ&hl" target="_blank">Yang&nbsp;Liu</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://jeremybernste.in" target="_blank">Jeremy&nbsp;Bernstein</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://www.bbe.caltech.edu/people/markus-meister" target="_blank">Markus&nbsp;Meister</a> &emsp; <b>&middot;</b> &emsp;
  <a href="http://www.yisongyue.com" target="_blank">Yisong&nbsp;Yue</a>
</p>

## Getting started

- Grab [nero.py](https://github.com/jxbz/nero/blob/master/nero.py) and place it in your Pytorch project directory. Then type:
```python
from nero import Nero
optimizer = Nero(net.parameters(), lr=0.01)
```
- An initial learning rate of `lr = 0.01` is the recommended default. This worked in almost all our experiments. Otherwise try `lr=0.001`. 
- Learning rate decay over the course of optimisation also helps.

## About this repository

This repository was built by <a href="https://scholar.google.com/citations?user=nVWQwHkAAAAJ&hl" target="_blank">Yang&nbsp;Liu</a> and <a href="https://jeremybernste.in" target="_blank">Jeremy&nbsp;Bernstein</a> to accompany the following paper:

> [Learning by Turning: Neural Architecture Aware Optimisation](https://arxiv.org/abs/2102.07227).

We're putting this code here so that you can test out our optimisation algorithm in your own applications, and also so that you can attempt to reproduce the experiments in our paper.

If something isn't clear or isn't working, let us know in the *Issues section* or contact [yang@abacus.ai](mailto:yang@abacus.ai) and [bernstein@caltech.edu](mailto:bernstein@caltech.edu).

## Repository structure

    .
    ├── cGAN/                   # Class conditional GAN image generation experiments
    ├── cifar/                  # CIFAR-10 classification experiments
    ├── imagenet/               # ImageNet classification experiments
    ├── mnist/                  # MNIST experiments with deep MLP and reparameterisation
    ├── optim/                  # optimiser definitions
    ├── ppo/                    # reinforcement learning experiment
    ├── transformer-wmt16/      # large transformer
    ├── wikitext2/              # small transformer
    ├── LICENSE                 # license on our algorithm
    ├── README.md               # the page you're reading now
    └── nero.py                 # our optimiser

## Citation

If you find Nero useful, feel free to cite [the paper](https://arxiv.org/abs/2102.07227):

```bibtex
@misc{nero2021,
  title={Learning by Turning: Neural Architecture Aware Optimisation},
  author={Yang Liu and Jeremy Bernstein and Markus Meister and Yisong Yue},
  year={2021},
  eprint={arXiv:2102.07227}
}
```

## License

We are making our algorithm available under a [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. The other code we have used obeys other license restrictions as indicated in the subfolders.
