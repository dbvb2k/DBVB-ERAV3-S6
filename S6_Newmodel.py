import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First block
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.1)
        
        # Second block
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(0.1)
        
        # MaxPool
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Third block
        self.conv3 = nn.Conv2d(16, 16, 3)
        self.bn3 = nn.BatchNorm2d(16)
        self.dropout3 = nn.Dropout(0.1)
        
        # Fourth block
        self.conv4 = nn.Conv2d(16, 16, 3)
        self.bn4 = nn.BatchNorm2d(16)
        self.dropout4 = nn.Dropout(0.1)
        
        # Fifth block
        self.conv5 = nn.Conv2d(16, 16, 3)
        self.bn5 = nn.BatchNorm2d(16)
        self.dropout5 = nn.Dropout(0.1)
        
        # Sixth block
        self.conv6 = nn.Conv2d(16, 16, 3)
        
        # Final 1x1 conv
        self.conv7 = nn.Conv2d(16, 10, 1)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        
        # Second block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        
        # MaxPool
        x = self.pool1(x)
        
        # Third block
        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        
        # Fourth block
        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn4(x)
        x = self.dropout4(x)
        
        # Fifth block
        x = self.conv5(x)
        x = F.relu(x)
        x = self.bn5(x)
        x = self.dropout5(x)
        
        # Sixth block
        x = self.conv6(x)
        
        # Final 1x1 conv
        x = self.conv7(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=1)

def get_model():
    return Net()

def save_model(model, accuracy):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mnist_model_{accuracy:.2f}_{timestamp}.pth"
    torch.save(model.state_dict(), f"models/{filename}")
    return filename
