import torch
import torch.nn as nn
import torch.nn.functional as F


class Model1(nn.Module):
    """
    Literally as basic as it gets
    """
    def __init__(self):
        super(Model1, self).__init__()
        
        self.fc1 = nn.Linear(28 * 28, 10)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
    
class Model2(nn.Module):
    """
    Model I tried based on previous work.
    Doesn't work that great.
    """
    def __init__(self):
        super(Model2, self).__init__()
        
        self.fc1 = nn.Linear(28, 64)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 32, 1)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 8, 1)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8 * 8 * 28, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.fc2(x)
        x = F.log_softmax(self.bn2(x), dim = 1)
        x = self.conv2(x)
        x = self.fc3(x)
        x = torch.flatten(x, 1)
        x = self.fc4(x)
        return x

class Model3(nn.Module):
    """
    Similar to model 1 with pooling + extra layers.
    """
    def __init__(self):
        super(Model3, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2, 1)
        self.fc1 = nn.Linear(225, 100)
        self.fc2 = nn.Linear(100, 40)
        self.fc3 = nn.Linear(40, 10)

    def forward(self, x):
        
        x = x.reshape(x.shape[0], 28, 28)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
class Model4(nn.Module):
    """
    Similar to model 3 with extra layers.
    Best working so far.
    """
    def __init__(self):
        super(Model4, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2, 1)
        self.fc1 = nn.Linear(225, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 100)
        self.fc4 = nn.Linear(100, 40)
        self.fc5 = nn.Linear(40, 10)

    def forward(self, x):
        
        x = x.reshape(x.shape[0], 28, 28)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        
        return x
