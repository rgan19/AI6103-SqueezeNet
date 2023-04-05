import argparse
import json
import os
import sys
import time
import warnings

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import binaryconnect
from models.squeezenet import SqueezeNet
from models.squeezenet import SqueezeNet
from preprocess import get_test_loader, get_train_valid_loader
from utils import plot_loss_acc, plotter, epoch_time

def train(train_loader, val_loader, model, criterion, optimizer, epochs, scheduler, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
      
    total_train_loss = []
    total_val_loss = []
    total_train_acc = []
    total_val_acc = []
    
    start_time = time.time()
    for epoch in range(epochs):
        train_acc, train_loss, val_acc, val_loss = 0, 0, 0, 0
        train_samples, val_samples = 0, 0 
        model.train()
        for images, labels in train_loader:
            batch_size = images.shape[0]

            images, labels = images.to(device), labels.to(device)

            if args.bc:
                binaryconnect.binarization()    # BC implementation  
            
            logits = model(images)
            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            
            if args.bc:
                binaryconnect.restore()     # BC implementation
                                  
            optimizer.step()
            
            if args.bc:
                binaryconnect.clip()     # BC implementation
            
            scheduler.step()
            
            _, top_class = logits.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            
            train_acc += torch.sum(equals).item()
            train_loss += batch_size * loss.item()
            train_samples += batch_size
            
        model.eval()
        for val_images, val_labels in val_loader:
            batch_size = val_images.shape[0]
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            
            val_logits = model(val_images)
            loss = criterion(val_logits, val_labels)
            
            _, top_class = val_logits.topk(1, dim=1)
            equals = top_class == val_labels.view(*top_class.shape)
            
            val_acc += torch.sum(equals).item()
            val_loss += batch_size * loss.item()
            val_samples += batch_size
            
        # print
        print(f"Epoch {(epoch+1):d}/{args.epochs:d}.. Learning rate: {scheduler.get_lr()[0]:.4f}.. Train loss: {(train_loss/train_samples):.4f}.. Train acc: {(train_acc/train_samples):.4f}.. Val loss: {(val_loss/val_samples):.4f}.. Val acc: {(val_acc/val_samples):.4f}")
        total_train_loss.append(train_loss/train_samples)
        total_train_acc.append(train_acc/train_samples)
        total_val_loss.append(val_loss/val_samples)
        total_val_acc.append(val_acc/val_samples)
        
    time_taken = epoch_time(start_time, time.time())
    print('Total Training Time', time_taken)
    return {
        "train_acc": total_train_acc,
        "train_loss": total_train_loss,
        "val_acc": total_val_acc,
        "val_loss": total_val_loss,
    }



def test(test_loader, model, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)    
    test_acc, test_loss = 0, 0
    test_samples = 0
    
    for test_images, test_labels in test_loader:
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        batch_size = test_images.shape[0]
        test_logits = model.forward(test_images)
        loss = criterion(test_logits, test_labels)
        _, top_class = test_logits.topk(1, dim=1)
        
        equals = top_class == test_labels.view(*top_class.shape)
        test_acc += torch.sum(equals).item()
        test_loss += batch_size * loss.item()        
        test_samples += batch_size

    return {
        "test_acc": test_acc / test_samples,
        "test_loss": test_loss / test_samples
    
    }
    
    
def parse_args(args):
    parser = argparse.ArgumentParser(description='SqueezeNet training parameter setting')
    parser.add_argument('--dataset_dir',type=str, help='')
    parser.add_argument('--fig_name',type=str, help='')
    parser.add_argument('--save_images', action='store_true', default=True)

    parser.add_argument('--epochs', default=10, type=int, metavar='N',
            help='number of total epochs to run')
    parser.add_argument('--test', action='store_true', default=False)
    
    parser.add_argument('-b', '--batch_size', default=128, type=int,
            help='mini-batch size (default: 128)')
    
    parser.add_argument('--mish', default=False, action='store_true', help='use mish activation function')
    parser.add_argument('--bc', '--binary_connect', default=False, action='store_true', help='apply binary connect')
    parser.add_argument('--skip', default=False, action='store_true', help='apply skip connection')
    
    parser.add_argument('--seed', type=int, default=0, help='set random seed value')
    parser.add_argument('--wd', default=0.0, type=float, help='weight decay')
    parser.add_argument('--lr', '--learning_rate', default=0.0003, type=float,
            metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr_scheduler', action='store_true', default=True, help='select learning rate schduler')
    parser.add_argument('--plot', default=False, action='store_true', help='plot graph')
    
    # TODO: add more arguments for different improvements 

    return parser.parse_args(args)


def main(args):
    print ('-- arguments -- \n', args)
    
    # Setup train data
    train_loader, val_loader, norm_value = get_train_valid_loader(args.dataset_dir, args.batch_size, True, args.seed, args.save_images)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print ('-- Model Setup')
    model = SqueezeNet()
    model.to(device)
    
    # BC implementation
    if args.bc:
        print('-binary connect')
        binaryconnect = binaryconnect.BC(model)

    print ('-- Criterion')
    criterion = torch.nn.CrossEntropyLoss().to(device)

    print ('-- Optimizer')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.epochs)

    print ('-- Training')
    train_result = train(train_loader, val_loader, model, criterion, optimizer, args.epochs, scheduler, args)
    print('Train result', train_result) # TODO: delete

    # save model results after training
    torch.save(model.state_dict(), './train-model.pt')
    np.savez(os.path.join('./diagram', args.fig_name.replace('.png ', '.npz')), train_loss=train_result['train_loss'], val_loss=train_result['val_loss'], train_acc=train_result['train_acc'], val_acc=train_result['val_acc'])

    # Run on test set
    if args.test:
        test_loader = get_test_loader(args.dataset_dir, args.batch_size, norm_value)
        test_result = test(test_loader, model, criterion)
        print('Test result', test_result)

    # plot
    if args.plot:
        plotter(train_result, args.fig_name)


if __name__ == '__main__':
  print('sys.arg', sys.argv[1:])
  args = parse_args(sys.argv[1:])
  main(args)
