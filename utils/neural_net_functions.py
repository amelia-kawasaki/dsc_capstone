import os
import copy
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

import utils.corruptions as cor

def train(net, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler = None, device = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    
    criterion = criterion.to(device)
    
    losses = []
    accs = []

    net.train()
        
    for epoch in range(num_epochs):
        
        total_loss = 0.0
        
        for data in train_loader:
            inputs, labels = data
            labels = labels.long()
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
                
        if scheduler is not None:
            scheduler.step()
            
        epoch_loss = total_loss / len(train_loader)
        losses.append(epoch_loss)
        
        if val_loader is not None:
            val_acc = accuracy(net, val_loader)
            accs.append(val_acc)

    return net, accs, losses

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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.pool1 = nn.AvgPool2d(2, 2)
        self.rep1 = nn.ReplicationPad1d((2, 1))
        self.fc1 = nn.Linear(17, 14)
        self.bn1 = nn.BatchNorm1d(14)
        self.fc2 = nn.Linear(196, 400)
        self.fc3 = nn.Linear(400, 800)
        self.bn2 = nn.BatchNorm1d(800)
        self.fc4 = nn.Linear(800, 200)
        self.fc5 = nn.Linear(200, 50)
        self.fc6 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = x.reshape(x.shape[0], 28, 28)
        x = self.pool1(x)
        x = self.rep1(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.bn2(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        
        return x

def run_label_corruption_nets(train_X, train_y, test_X, test_y, num_labels, benchmarks, corruption_levels, num_iters = 5):
    
    train_X, train_y, test_X, test_y = torch.Tensor(train_X), torch.Tensor(train_y), torch.Tensor(test_X), torch.Tensor(test_y)
    testing_dataset = TensorDataset(test_X, test_y)
    
    test_dataloader = DataLoader(testing_dataset, batch_size = 100, shuffle = True)
    
    test_accuracies = {}
    for bm in benchmarks:
        test_accuracies[bm] = []
    
    for i in range(num_iters):
        
        for bm in benchmarks:
            test_accuracies[bm].append([])
            
        for c in corruption_levels:
            
            corrupted_y = cor.label_randomizer(copy.deepcopy(train_y), c)
            training_dataset = TensorDataset(train_X, corrupted_y)
            train_dataloader = DataLoader(training_dataset, batch_size = 100, shuffle=True)
            
            net = Model()
    
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.RMSprop(net.parameters(), lr=0.00001, momentum=0.9)
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00000000001, max_lr=0.0001,step_size_up=5,mode="triangular2")
            
            for k in range(len(benchmarks)):
                if k == 0:
                    net, accs, los = train(net, train_dataloader, None, benchmarks[k], criterion, optimizer, scheduler)
                else:
                    net, accs, los = train(net, train_dataloader, None, benchmarks[k] - benchmarks[k - 1], criterion, optimizer, scheduler)
                    
                acc = accuracy(net, test_dataloader)
                test_accuracies[benchmarks[k]][i].append(acc)
                
    return test_accuracies

def run_random_corruption_nets(train_X, train_y, test_X, test_y, num_labels, benchmarks, corruption_levels, num_iters = 5):
    
    train_X, train_y, test_X, test_y = torch.Tensor(train_X), torch.Tensor(train_y), torch.Tensor(test_X), torch.Tensor(test_y)
    testing_dataset = TensorDataset(test_X, test_y)
    
    test_dataloader = DataLoader(testing_dataset, batch_size = 100, shuffle = True)
    
    test_accuracies = {}
    for bm in benchmarks:
        test_accuracies[bm] = []
    
    for i in range(num_iters):
        
        for bm in benchmarks:
            test_accuracies[bm].append([])
            
        for c in corruption_levels:
            
            corrupted_X = cor.random_filter(copy.deepcopy(train_X), c)
            training_dataset = TensorDataset(corrupted_X, train_y)
            train_dataloader = DataLoader(training_dataset, batch_size = 100, shuffle=True)
            
            net = Model()
    
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.RMSprop(net.parameters(), lr=0.00001, momentum=0.9)
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00000000001, max_lr=0.0001,step_size_up=5,mode="triangular2")
            
            for k in range(len(benchmarks)):
                if k == 0:
                    net, accs, los = train(net, train_dataloader, None, benchmarks[k], criterion, optimizer, scheduler)
                else:
                    net, accs, los = train(net, train_dataloader, None, benchmarks[k] - benchmarks[k - 1], criterion, optimizer, scheduler)
                    
                acc = accuracy(net, test_dataloader)
                test_accuracies[benchmarks[k]][i].append(acc)
                
    return test_accuracies
            