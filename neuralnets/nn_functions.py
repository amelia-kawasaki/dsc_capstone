import torch
import time
import numpy as np
from tqdm import tqdm

from torch.optim.lr_scheduler import StepLR

def train(net, train_loader, num_epochs, criterion, optimizer, scheduler = None, device = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    
    criterion = criterion.to(device)

    net.train()
    
    total_time = 0.0
    
    for epoch in range(num_epochs):
        
        t = time.time()
        total_loss = 0.0
        
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
                
        if scheduler is not None:
            scheduler.step()
                
        time_taken = time.time() - t
        print(f'End of epoch {epoch + 1} ({round(time_taken, 1)}s)   loss {total_loss / len(train_loader)}')
        total_time += time_taken

    print('\nFinished Training')
    print(f'Time taken: {round(total_time / 60, 2)} minutes')

    return net

def train_with_progress(net, train_loader, num_epochs, criterion, optimizer, scheduler, device = None):
    """
    Does the same thing as train(), however this one prints a fancy progress bar.
    Provides a visualiztion of progress at the cost of taking extra time.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    criterion = criterion.to(device)

    net.train()
    
    total_time = 0
    for epoch in range(num_epochs):
        
        with tqdm(train_loader, unit="batch") as tepoch:
            t = time.time()
            total_loss = 0.0
        
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                
                tepoch.set_postfix(loss=loss.item())

                total_loss += loss.item()
                
        if scheduler is not None:
            scheduler.step()
                
        print(f'Total loss {total_loss / len(train_loader)}')

    return net

def accuracy(net, test_loader, path = None, device = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    
    num_correct = 0
    total = 0
    
    net.eval()
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)
            
            num_correct += int(sum(preds == labels))
            total += len(labels)
            
    if path is not None:
        torch.save(net, path)
            
    return num_correct / total
