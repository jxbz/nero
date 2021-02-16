import itertools
import sys
# from tqdm import tqdm
tqdm = lambda x: x
import math

import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
from torchvision import datasets, transforms

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

def CWN(W):
    centered = W - W.mean(dim=1, keepdim=True)
    return centered / centered.norm(dim=1, keepdim=True)

class CWNNet(nn.Module):
    def __init__(self, depth, width):
        super(CWNNet, self).__init__()
        
        self.initial = nn.Linear(784, width, bias=False)
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(depth-2)])
        self.final = nn.Linear(width, 10, bias=False)
        
    def forward(self, x):
        cwn = CWN(self.initial.weight)
        x = torch.mm(x, cwn.t())
        x = F.relu(x) * math.sqrt(2)
        for layer in self.layers:
            cwn = CWN(layer.weight)
            x = torch.mm(x, cwn.t())
            x = F.relu(x) * math.sqrt(2)
        cwn = CWN(self.final.weight)
        x = torch.mm(x, cwn.t())
        return x
    
def getLR(optimizer):
    for param_group in optimizer.param_groups:
        print(f"lr is {param_group['lr']}")

## Define a function to train a network

def train_network(depth, width, epochs, init_lr, init_scale, decay, seed):
    
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    def init_weights(m):
        if type(m) == nn.Linear:
            print("reinit", type(m), "on scale", init_scale)
            torch.nn.init.normal_(m.weight, mean=0.0, std=init_scale)

    model = CWNNet(depth, width).cuda()
    model.apply(init_weights)

    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=init_lr)
      
    lr_lambda = lambda x: decay**x
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    print("\n==========================================================")
    print(f"Training {depth} layers, width {width}, optim {type(optim).__name__}, initial scale {init_scale}, seed {seed}\n")

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


            this_correct = (target == y_pred.max(dim=1)[1]).sum().item()
            this_total = target.shape[0]
            train_acc_list.append(this_correct/this_total)
            correct += this_correct
            total += this_total

            model.zero_grad()
            loss.backward()
            optim.step()

        lr_scheduler.step()
        print(f"Epoch {epoch} train_acc {correct/total} lr is {optim.param_groups[0]['lr']}")

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

## Train with different initialisation scales

width = 784
depth = 5

epochs = 5
decay = 1.0
init_lr = 0.01

seed_list = [0,1,2]
init_scale_list = [1,100]
results = {}

for init_scale, seed in itertools.product(init_scale_list, seed_list):
    train_acc, test_acc = train_network(  depth=depth, width=width, epochs=epochs, init_lr=init_lr, init_scale=init_scale, decay=decay, seed=seed )
    print( f'\nFinal test acc: {test_acc}' )

    results[init_scale, seed] = train_acc
    with open('results/cwn_results.pickle', 'wb') as handle:
        pickle.dump(results, handle)