import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from cifar import CIFAR100

parser = argparse.ArgumentParser()
# Directory
parser.add_argument('--data_path', default='./data')
parser.add_argument('--save_path', default='./save')
# Hyperparameters
parser.add_argument('--noise_fraction', default=0.3)
parser.add_argument('--max_epoch', default=200)
parser.add_argument('--batchsize', default=256)
parser.add_argument('--lr', default=0.001)
parser.add_argument('--weight_decay', default=0.0001)
parser.add_argument('--lr_decay_interval', default=20)
# Utility parameters
parser.add_argument('--log_interval', default=50)
parser.add_argument('--num_workers', default=16)
parser.add_argument('--gpu', default=1)
parser.add_argument('--baseline', default=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(8*8*128, 500),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(300, 100)
        )
    
    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 8*8*128)
        x = self.classifier(x)
        return x

class NoiseLayer(nn.Module):
    def __init__(self, theta, k=100):
        super(NoiseLayer, self).__init__()
        self.theta = nn.Linear(k, k, bias=False)
        self.theta.weight.data = nn.Parameter(theta)
        self.eye = torch.Tensor(np.eye(k))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        theta = self.eye.to(x.device).detach()
        theta = self.theta(theta)
        theta = self.softmax(theta)
        out = torch.matmul(x, theta)
        return out


def main():
    opt = parser.parse_args()
    trainset = CIFAR100(opt.data_path,  train=True,
                                        transform=transforms.Compose([
                                            # transforms.RandomHorizontalFlip(),
                                            # transforms.RandomCrop(32, padding=4),
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                # (0.2023, 0.1994, 0.2010))
                                        ]),
                                        download=True,
                                        noise_fraction=opt.noise_fraction)
    testset = CIFAR100(opt.data_path,  train=False,
                                        transform=transforms.Compose([
                                            # transforms.RandomHorizontalFlip(),
                                            # transforms.RandomCrop(32, padding=4),
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                # (0.2023, 0.1994, 0.2010))
                                        ]),
                                        download=True,
                                        noise_fraction=opt.noise_fraction)

    trainloader = data.DataLoader(trainset, batch_size=opt.batchsize, 
                                            shuffle=True, 
                                            num_workers=opt.num_workers)
    testloader = data.DataLoader(testset, batch_size=opt.batchsize, 
                                            shuffle=False, 
                                            num_workers=opt.num_workers)
    
    epsilon = 0.00001
    theta = torch.Tensor((1-epsilon) * np.eye(100) + epsilon/(100-1) * (1-np.eye(100)))

    model = CNN().cuda(opt.gpu)
    noisemodel = NoiseLayer(theta=theta.cuda(opt.gpu))

    criterion = nn.CrossEntropyLoss().cuda(opt.gpu)        
    optimizer = optim.Adam(model.parameters(), 
                            lr=opt.lr, 
                            weight_decay=opt.weight_decay)
    noise_optimizer = optim.Adam(noisemodel.parameters(), 
                            lr=opt.lr, 
                            weight_decay=opt.weight_decay)


    best_acc = 0.0
    for epoch in range(opt.max_epoch):
        if epoch > 30:
            adjust_learning_rate(optimizer, epoch, opt)
        train(trainloader, model, noisemodel, optimizer, noise_optimizer, criterion, epoch, opt)
        acc = test(testloader, model, criterion, opt)
        if acc > best_acc:
            best_acc = acc
            # state = {
            #     'state_dict':model.state_dict(),
            #     'optimizer':optimizer.state_dict()
            # }
            # torch.save(state, os.path.join(opt.save_path, '%d_checkpoint.ckpt'))
        print(' Best accuracy so far : %.4f%%'%best_acc)


def train(train_loader, model, noisemodel, optimizer, noise_optimizer, criterion, epoch, opt):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    model.train()
    for i, (x, target, noisy_target) in enumerate(tqdm(train_loader, desc='##Training CIFAR100')):
        x = x.cuda(opt.gpu)
        target = target.cuda(opt.gpu)
        noisy_target = noisy_target.cuda(opt.gpu)

        out = model(x)
        if not opt.baseline:
            out = noisemodel(out)
        loss = criterion(out, noisy_target)

        optimizer.zero_grad()
        if not opt.baseline:
            noise_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not opt.baseline:
            noise_optimizer.step()

        acc = accuracy(out, noisy_target)
        acc_meter.update(acc, out.size(0))
        loss_meter.update(loss, out.size(0))

        if i%opt.log_interval==0:
            string = '[%d/%d][%d/%d] Loss : %.4f / %.4f, Acc : %.4f%% / %.4f%%'% \
                (epoch, opt.max_epoch, i, len(train_loader), loss_meter.val, loss_meter.avg, 
                acc_meter.val*100.0, acc_meter.avg*100.0)
            tqdm.write(string)

def test(test_loader, model, criterion, opt):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (x, target, _) in enumerate(tqdm(test_loader, desc='##Evaluating CIFAR100')):
            x = x.cuda(opt.gpu)
            target = target.cuda(opt.gpu)

            out = model(x)
            loss = criterion(out, target)

            acc = accuracy(out, target)
            acc_meter.update(acc, out.size(0))
            loss_meter.update(loss, out.size(0))
    print('# Evaluation || Loss : %.4f, Acc : %.4f%%'% (loss_meter.avg, acc_meter.avg*100.0))
    return acc_meter.avg*100.0

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, opt):
    lr = opt.lr * (0.3 ** ((epoch - 30) // opt.lr_decay_interval))
    print('Learning rate : %f'%lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target):
    correct = 0
    _, pred = output.max(1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    acc = correct/target.size(0)
    return acc

if __name__=='__main__':
    main()