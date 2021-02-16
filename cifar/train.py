# train.py
#!/usr/bin/env	python3

""" train network using pytorch

Adapted by Yang Liu, original repo:
https://github.com/weiaicunzai/pytorch-cifar100

"""

import os
import sys
import argparse
import time
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR

import math
import sys
sys.path.append('../optim')

from nero import Nero,Nero_abl
from lamb import Lamb
from lambcs import LambCS
from madam import Madam
from madamcs import MadamCS

def train(epoch):
    pf = 0 #False
    start = time.time()
    net.train()

    for batch_index, (images, labels) in enumerate(cifar_training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()

        optimizer.step()

        n_iter = (epoch - 1) * len(cifar_training_loader) + batch_index + 1

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar_training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(dataloader=None, train=False, epoch=None):

    start = time.time()
    net.eval()

    test_loss = 0.0 
    correct = 0.0

    for (images, labels) in dataloader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item() * len(labels)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print('Evaluating Network.....')
    acc = correct.float() / len(dataloader.dataset)
    mean_loss = test_loss / len(dataloader.dataset)

    name = "train" if train else "test"
    print('{} set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        name, mean_loss, acc, finish - start ))

    #add informations to tensorboard
    if train:
        writer.add_scalar('Train/Average loss', mean_loss, epoch)
        writer.add_scalar('Train/Accuracy', acc, epoch)
    else:
        writer.add_scalar('Test/Average loss', mean_loss, epoch)
        writer.add_scalar('Test/Accuracy', acc, epoch)

    return acc, mean_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--net', type=str, required=True, help='net type')
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('--b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--warm', type=int, default=5, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    # Yang added:
    parser.add_argument('--momentum', default=0.0 , type=float,help='momentum/beta1')
    parser.add_argument('--beta', default=0.999, type=float,help='beta2')

    parser.add_argument('--prefix', default='', type=str)
    parser.add_argument('--gamma', default=0.2 , type=float)
    parser.add_argument('--wd', default=0.0005 , type=float)

    parser.add_argument('--c1', action='store_true', dest="c1", default=False, help='Nero mean constraint')
    parser.add_argument('--c2', action='store_true', dest="c2", default=False, help='Nero norm constraint')

    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--task', default='cifar10' , type=str)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    net = get_network(args)

    #data preprocessing:
    if args.task == "cifar100":
        mean = settings.CIFAR100_TRAIN_MEAN
        std = settings.CIFAR100_TRAIN_STD
    elif args.task == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        print("invalid task!!")

    cifar_training_loader = get_training_dataloader(
        mean,
        std,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        alpha = 0.0,
        task = args.task,
        da = True
    )

    cifar_test_loader = get_test_dataloader(
        mean,
        std,
        num_workers=4,
        batch_size=args.b,
        shuffle=False,
        task = args.task
    )
    #test training acc
    cifar_train_test_loader = get_test_dataloader(
        mean,
        std,
        num_workers=4,
        batch_size=args.b,
        shuffle=False,
        task = args.task,
        train = True
    )

    loss_function = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        print("using sgd!")
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    
    elif args.optimizer == 'adam':
        print("using adam!")
        optimizer = optim.Adam(net.parameters(), lr=args.lr,betas=(args.momentum, args.beta), weight_decay=args.wd)
    
    elif args.optimizer == 'lamb':
        print("using lamb!")
        optimizer = Lamb(net.parameters(), lr=args.lr,betas=(args.momentum, args.beta), weight_decay=args.wd)
    elif args.optimizer == 'lambcs':
        print("using lambcs!")
        optimizer = LambCS(net.parameters(), lr=args.lr,betas=(args.momentum, args.beta), weight_decay=args.wd,
                            constraints=True)

    elif args.optimizer == 'madam':
        print("using madam!")
        optimizer = Madam(net.parameters(), lr=args.lr)

    elif args.optimizer == 'madamcs':
        print("using madamcs!")
        optimizer = MadamCS(net.parameters(), lr=args.lr,constraints=True)
   
    elif args.optimizer == 'nero':
        print("using nero!")
        optimizer = Nero(net.parameters(),lr=args.lr,constraints=True)

    elif args.optimizer == 'neroabl':
        print("using nero ablated!")
        optimizer = Nero_abl(net.parameters(),lr=args.lr,
                        c1=args.c1,c2=args.c2)
    
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=args.gamma) #learning rate decay
    iter_per_epoch = len(cifar_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    args.prefix = "seed" + str(args.seed) + args.prefix 
    
    if args.optimizer == "sgd":
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.task, args.net, 
                        args.prefix + args.optimizer + str(args.lr)+'_g_'+str(args.gamma)+
                        '_momentum_'+str(args.momentum)+'_wd_'+str(args.wd),
                        settings.TIME_NOW)
    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.task, args.net, 
                        args.prefix + args.optimizer + str(args.lr)+'_g_'+str(args.gamma)+
                        '_beta1_'+str(args.momentum)+'_beta2_'+str(args.beta)+'_wd_'+str(args.wd),
                        settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    if args.optimizer == "sgd":
        writer = SummaryWriter(log_dir=os.path.join(
                            settings.LOG_DIR, args.task, args.net,
                            args.prefix + args.optimizer + str(args.lr)+'_g_'+str(args.gamma)+
                            '_momentum_'+str(args.momentum)+'_wd_'+str(args.wd),
                            settings.TIME_NOW))
    else:
        writer = SummaryWriter(log_dir=os.path.join(
                            settings.LOG_DIR, args.task, args.net,
                            args.prefix + args.optimizer + str(args.lr)+'g'+str(args.gamma)+
                            '_beta1_'+str(args.momentum)+'_beta2_'+str(args.beta)+'_wd_'+str(args.wd),
                            settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32).cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    best_test_acc = 0.0
    best_test_acc_epoch = 0

    best_test_loss = 10.0
    best_test_loss_epoch = 0
        
    best_train_acc = 0.0
    best_train_acc_epoch = 0

    best_train_loss = 10.0
    best_train_loss_epoch = 0

    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        writer.add_scalar("lr",optimizer.param_groups[0]['lr'],epoch)
        
        train(epoch)

        test_acc, test_loss = eval_training(dataloader=cifar_test_loader,train=False,epoch=epoch)
        train_acc, train_loss = eval_training(dataloader=cifar_training_loader,train=True,epoch=epoch)
        print(writer.log_dir)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_acc_epoch = epoch

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_loss_epoch = epoch
        
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_train_acc_epoch = epoch

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_loss_epoch = epoch

        #start to save best performance model after learning rate decay to 0.01
        writer.add_scalar('Test/Best Average loss', best_test_loss, epoch)
        writer.add_scalar('Test/Best Accuracy', best_test_acc, epoch)
        writer.add_scalar('Train/Best Average loss', best_train_loss, epoch)
        writer.add_scalar('Train/Best Accuracy', best_train_acc, epoch)


        if epoch > settings.MILESTONES[1] and best_acc < test_acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = test_acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    writer.close()
