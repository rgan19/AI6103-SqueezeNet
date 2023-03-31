import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


def train(train_loader, val_loader, model, criterion, optimizer, epochs, scheduler, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    
    total_train_loss = []
    total_val_loss = []
    total_train_acc = []
    total_val_acc = []
    
    for epoch in range(epochs):
        train_acc, train_loss, val_acc, val_loss = 0, 0, 0, 0
        train_samples, val_samples = 0, 0 
        model.train()
        for images, labels in train_loader:
            batch_size = images.shape[0]

            images, labels = images.to(device), labels.to(device)
            logits = model(images)

            optimizer.zero_grad()

            loss = criterion(logits, labels) 
            
            loss.backward()
            optimizer.step()
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
            
            val_acc += torch.sum(equals)
            val_loss += batch_size * loss.item()
            val_samples += batch_size
            
        total_train_loss.append(train_loss/train_samples)
        total_train_acc.append(train_acc/train_samples)
        total_val_loss.append(val_loss/val_samples)
        total_val_acc.append(val_acc/val_samples)
        
    
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
    
    
    