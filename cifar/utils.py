""" helper function

author baiyu
"""

import sys

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# Yang
import cifar

def get_network(args):
    """ return given network
    """
    if args.task == 'cifar10':
        nclass = 10
    elif args.task == 'cifar100':
        nclass = 100
    #Yang added none bn vggs
    if args.net == 'vgg11':
        from models.vgg import vgg11
        net = vgg11(num_classes=nclass)
    elif args.net == 'vgg13':
        from models.vgg import vgg13
        net = vgg13(num_classes=nclass)
    elif args.net == 'vgg16':
        from models.vgg import vgg16
        net = vgg16(num_classes=nclass)
    elif args.net == 'vgg19':
        from models.vgg import vgg19
        net = vgg19(num_classes=nclass) 
    
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_classes=nclass)
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(num_classes=nclass)
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_classes=nclass)
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(num_classes=nclass)
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(num_classes=nclass)

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True,alpha=0.0,task='cifar100',da=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    if da:
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else: # disable data augmentaion
        print("no data augmentation!")
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if task == 'cifar100':
        cifar100_training = cifar.CIFAR100(root='./data', train=True, download=True, transform=transform_train,alpha=alpha)
    elif task == 'cifar10':
        cifar100_training = cifar.CIFAR10(root='./data', train=True, download=True, transform=transform_train,alpha=alpha)
    
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size,drop_last=False)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True,task="cifar100",train=False):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    if task == "cifar100":
        cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform_test)
    elif task == "cifar10":
        cifar100_test = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform_test)
    
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
