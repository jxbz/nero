import itertools
import sys
# from tqdm import tqdm
tqdm = lambda x: x
import math

import argparse
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
from torchvision import datasets, transforms

import sys
sys.path.append('..')
from optim.nero import Nero
from optim.nero import neuron_norm
from optim.nero import neuron_mean
from optim.lamb import Lamb

## Get the data and put it in a Pytorch dataloader

batch_size = 250

trainset = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

testset = datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

## Define the multilayer perceptron architecture

class SimpleNet(nn.Module):
    def __init__(self, depth, width, bias):
        super(SimpleNet, self).__init__()
        
        self.initial = nn.Linear(784, width, bias=bias)
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=bias) for _ in range(depth-2)])
        self.final = nn.Linear(width, 10, bias=bias)
        
    def forward(self, x):
        x = self.initial(x)
        x = F.relu(x) * math.sqrt(2)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x) * math.sqrt(2)
        return self.final(x)
    
def getLR(optimizer):
    for param_group in optimizer.param_groups:
        print(f"lr is {param_group['lr']}")

## Define a function to train a network

def train_network(depth, width, bias, epochs, opt, init_lr, decay, seed):
    
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    model = SimpleNet(depth, width, bias).cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    
    if opt == "nero":
        optim = Nero(model.parameters(), lr=init_lr)
    elif opt == "lamb":
        optim = Lamb(model.parameters(), lr=init_lr)
    elif opt == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=init_lr)
    elif opt == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=init_lr)
      
    lr_lambda = lambda x: decay**x
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    print("\n==========================================================")
    print(f"Training {depth} layers, width {width}, optim {type(optim).__name__}, initial learning rate {init_lr}, seed {seed}\n")

    model.train()

    train_acc_list = []

    for epoch in range(epochs):

        correct = 0
        total = 0

        for data, target in tqdm(train_loader):
            data, target = (data.cuda(), target.cuda())

            data = data.view(batch_size,-1)
            y_pred = model(data)
            loss = loss_fn(y_pred, target)

            correct += (target == y_pred.max(dim=1)[1]).sum().item()
            total += target.shape[0]

            model.zero_grad()
            loss.backward()
            optim.step()

        lr_scheduler.step()
        print(f"Epoch {epoch} train_acc {correct/total} lr is {optim.param_groups[0]['lr']}")
        train_acc_list.append(correct/total)

    model.eval()
    correct = 0
    total = 0

    for data, target in tqdm(test_loader):
        data, target = (data.cuda(), target.cuda())

        data = data.view(batch_size,-1)
        y_pred = model(data)
        loss = loss_fn(y_pred, target)

        correct += (target == y_pred.max(dim=1)[1]).sum().item()
        total += target.shape[0]

    test_acc = correct/total
  
    return train_acc_list, test_acc

## Set architecture and training hyperparams

width = 784
depth = 100
bias = True

epochs = 50
decay = 0.9

parser = argparse.ArgumentParser(description='100 layer MLP')
parser.add_argument('--lr', action='store_true', help='whether to tune LR')
parser.add_argument('--seeds', action='store_true', help='whether to run three seeds')
args = parser.parse_args()

if args.lr:
    print("Running LR search")

    opt_list = ['nero', 'lamb', 'adam', 'sgd']
    init_lr_list = [0.0001, 0.001, 0.01, 0.1]

    seed = 0

    results = {}
    for opt, init_lr in itertools.product(opt_list, init_lr_list):
        train_acc, test_acc = train_network(  depth=depth, width=width, bias=bias, epochs=epochs, opt=opt, init_lr=init_lr, decay=decay, seed=seed )
        print( f'\nFinal test acc: {test_acc}' )
        
        results[opt, init_lr, seed] = train_acc, test_acc
        pickle.dump( results, open( "results/100-layer-lr-search.p", "wb" ) )

if args.seeds:
    print("Running three random seeds")

    opt_lr_list = [('nero', 0.001), ('lamb', 0.01), ('adam', 0.001), ('sgd', 0.1)]
    seed_list = [0, 1, 2]

    results = {}
    for (opt, init_lr), seed in itertools.product(opt_lr_list, seed_list):
        train_acc, test_acc = train_network(  depth=depth, width=width, bias=bias, epochs=epochs, opt=opt, init_lr=init_lr, decay=decay, seed=seed )
        print( f'\nFinal test acc: {test_acc}' )
        
        results[opt, init_lr, seed] = train_acc, test_acc
        pickle.dump( results, open( "results/100-layer-3-seeds.p", "wb" ) )
