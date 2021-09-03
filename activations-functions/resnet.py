from __future__ import print_function
import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torch.nn import utils
from torch.optim import lr_scheduler
import torch.utils.data.dataloader as dataloader
from torchvision import datasets, transforms
from torch import autograd
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

import numpy as np


def Variable(x, **kwargs):
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        return autograd.Variable(x, **kwargs).cuda()
    else:
        return autograd.Variable(x, **kwargs)


parser = argparse.ArgumentParser(description='PyTorch ResNnet MNIST Training')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')

parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument("--lrGamma", default=0.1, type=float)

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--name')
parser.add_argument('--nonlin')

SEED = 24
CUDA = False

if torch.cuda.is_available():
    CUDA = True

args = parser.parse_args()
experiment = args.name + "-" + args.nonlin
best_prec1 = 0
step = 0
logger = SummaryWriter(log_dir="/home/jose/empnonlin/src/runsFinal/" + experiment + "/", comment=experiment)


def main():
    nonlins = {
        "relu": F.relu,
        "elu": F.elu,
        "leaky_relu": F.leaky_relu,

        "tanh": F.tanh,
        "sigmoid": F.sigmoid,
    }

    assert args.nonlin in nonlins, "Did not recognize the provided nonlin."

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    global best_prec1
    global step

    # create model
    model = ResNet(nonlins[args.nonlin], IdentityBlock, ConvolutionalBlock)
    if CUDA:
        model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lrGamma)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_loader = torch.utils.data.DataLoader(datasets.MNIST(
        args.data,
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ), batch_size=args.batch_size, num_workers=2, shuffle=True, pin_memory=CUDA)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST(
        args.data,
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ), batch_size=args.batch_size, num_workers=2, shuffle=True, pin_memory=CUDA)

    if args.evaluate:
        validate(test_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        loss = train(train_loader, model, criterion, optimizer, epoch, args.epochs)

        ## evaluate on validation set
        prec1 = validate(test_loader, model, criterion)
        logger.add_scalar('data/(test)prec1', prec1, epoch + 1)

        ## remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, name=args.nonlin)

        # decay the LR
        scheduler.step(loss, epoch=epoch)
        mean_lr = np.mean([float(p['lr']) for p in optimizer.param_groups])
        logger.add_scalar('data/learning_rate', mean_lr, epoch + 1)

    logger.close()


def getLayerWeights(x):
    return x.weight.data.cpu()


def make_conv_grid(layer, num_cols=6):
    tensor = layer.weight.data.cpu().numpy()
    if not tensor.ndim == 4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1] == 3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1 + num_kernels // num_cols

    filters = []
    for i in range(tensor.shape[0]):
        filters.append(tensor[i])

    return make_grid(filters)


def train(train_loader, model, criterion, optimizer, epoch, num_epochs):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    global step
    start = time.time()
    # pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - start)

        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        batchTime = time.time()
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - batchTime)

        step += 1
        if i % args.print_freq == 0 and i > 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    # log the layers and layers gradient histogram and distributions
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.add_histogram('model/(train)' + tag, to_np(value).flatten(), step + 1)

    logger.add_scalar('data/(train)loss_val', losses.val, step)
    logger.add_scalar('data/(train)loss_avg', losses.avg, step)

    logger.add_scalar('data/(train)prec1', top1.avg, step)
    logger.add_scalar('data/(train)prec5', top5.avg, step)

    return losses.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input_var = Variable(input, volatile=True)
        target_var = Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, name='', filename='checkpoint.pth.tar'):
    if name != '':
        filename = name + '_' + filename

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, name + '_' + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def to_np(x):
    """
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/main.py#L20
    :param x:
    :return:
    """
    return x.data.cpu().numpy()


class IdentityBlock(nn.Module):
    def __init__(self, nonlin, in_channel, out_channels, kernel_size):
        super(IdentityBlock, self).__init__()
        F1, F2, F3 = out_channels

        # First component of main path
        self.conv1 = nn.Conv2d(in_channel, F1, kernel_size=1, stride=1)  # (28-7+2*3)/2 +1 = 15
        self.bn1 = nn.BatchNorm2d(F1)
        self.nonlin = nonlin

        # Second component of main path
        self.conv2 = nn.Conv2d(F1, F2, kernel_size, stride=1, padding=1)  # padding = f-1/2 = 3 -1/2
        self.bn2 = nn.BatchNorm2d(F2)

        # Third component of main path
        self.conv3 = nn.Conv2d(F2, F3, kernel_size=1, stride=1)  # F3 = 256
        self.bn3 = nn.BatchNorm2d(F3)

    def forward(self, x):
        residual = x

        # First component
        out = self.nonlin(self.bn1(self.conv1(x)))

        # Second component

        out = self.nonlin(self.bn2(self.conv2(out)))

        # Third component
        out = self.nonlin(self.bn3(self.conv3(out)))

        # Final step:
        out = out + residual
        out = self.nonlin(out)

        # print(out)
        return out


class ConvolutionalBlock(nn.Module):
    def __init__(self, nonlin, in_channel, out_channels, kernel_size, stride):
        super(ConvolutionalBlock, self).__init__()
        F1, F2, F3 = out_channels
        # First component of main path
        self.conv1 = nn.Conv2d(in_channel, F1, kernel_size=1, stride=stride)  # (28-7+2*3)/2 +1 = 15
        self.bn1 = nn.BatchNorm2d(F1)
        self.nonlin = nonlin

        # Second component of main path
        self.conv2 = nn.Conv2d(F1, F2, kernel_size, stride=1, padding=1)  # padding = f-1/2 = 3 -1/2
        self.bn2 = nn.BatchNorm2d(F2)
        # Third component of main path

        self.conv3 = nn.Conv2d(F2, F3, kernel_size=1, stride=1)  # F3 = 256
        self.bn3 = nn.BatchNorm2d(F3)

        ###### SHORTCUT PATH ####

        self.conv4 = nn.Conv2d(in_channel, F3, kernel_size=1, stride=stride)
        self.bn4 = nn.BatchNorm2d(F3)

    def forward(self, x):
        residual = x
        # First component
        out = self.nonlin(self.bn1(self.conv1(x)))

        # Second component
        out = self.nonlin(self.bn2(self.conv2(out)))
        # Third component
        out = self.nonlin(self.bn3(self.conv3(out)))
        ##### SHORTCUT PATH ####
        residual = self.bn4(self.conv4(residual))

        # Final step:
        out = out + residual
        out = self.nonlin(out)
        #   print(out)
        return out


class ResNet(nn.Module):
    def __init__(self, nonlin, id_block, conv_block, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16

        # out_channels =  [64, 64, 256]
        # F1, F2, F3 = filters
        # Stage 1 changed kernel size=3
        self.conv = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=3)  # (32-7+2*3)/2 +1 = 33
        self.bn = nn.BatchNorm2d(8)
        self.nonlin = nonlin

        # Stage 2
        st2_out_channels = [8, 8, 32]

        self.conv2 = self.make_layer_convolutional(conv_block, 8, st2_out_channels, 3, 1)
        self.id21 = self.make_layer_identity(id_block, 32, st2_out_channels, 3)
        self.id22 = self.make_layer_identity(id_block, 32, st2_out_channels, 3)

        # Stage 3 # Stage 1 changed the stride = 1
        st3_out_channels = [16, 16, 64]

        self.conv3 = self.make_layer_convolutional(conv_block, 32, st3_out_channels, 3, 1)
        self.id31 = self.make_layer_identity(id_block, 64, st3_out_channels, 3)
        self.id32 = self.make_layer_identity(id_block, 64, st3_out_channels, 3)
        self.id33 = self.make_layer_identity(id_block, 64, st3_out_channels, 3)

        # Final
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

        self.initParameters()

    def initParameters(self):
        for p in self.parameters():
            if p.dim() == 1:
                init.constant(p.data, 0.1)

            if p.dim() > 1:
                init.kaiming_normal(p.data, 0.25)

    def make_layer_identity(self, id_block, in_channel, out_channels, kernel_size):
        layers = []
        layers.append(id_block(self.nonlin, in_channel, out_channels, kernel_size))
        return nn.Sequential(*layers)

    def make_layer_convolutional(self, conv_block, in_channel, out_channels, kernel_size, stride):
        layers = []
        layers.append(conv_block(self.nonlin, in_channel, out_channels, kernel_size, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stage 1
        out = F.max_pool2d(self.nonlin(self.bn(self.conv(x))), kernel_size=3, stride=2)

        # Stage 2
        out = self.conv2(out)
        out = self.id21(out)
        out = self.id22(out)

        # Stage 3
        out = self.conv3(out)
        out = self.id31(out)
        out = self.id32(out)
        out = self.id33(out)

        # Final Step
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


if __name__ == '__main__':
    main()
