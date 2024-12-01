from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

# First model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First Convolution Block
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)  # input: 28x28x1, output: 26x26x8
        
        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)  # output: 24x24x16
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.05)
        
        # First MaxPooling
        self.pool1 = nn.MaxPool2d(2, 2)  # output: 12x12x16
        
        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3)  # output: 10x10x16
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(0.1)
        
        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3)  # output: 8x8x16
        self.bn3 = nn.BatchNorm2d(16)
        self.dropout3 = nn.Dropout(0.2)
        
        # Second MaxPooling
        self.pool2 = nn.MaxPool2d(2, 2)  # output: 4x4x16
        
        # Fifth Convolution Block
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3)  # output: 2x2x16
        
        # Final Convolution Block
        self.conv6 = nn.Conv2d(16, 10, kernel_size=2)  # output: 1x1x10
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # Added GAP layer

    def forward(self, x):
        # First Convolution Block
        x = F.relu(self.conv1(x))
        
        # Second Convolution Block
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        
        # First MaxPooling
        x = self.pool1(x)
        
        # Third Convolution Block
        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        
        # Fourth Convolution Block
        x = F.relu(self.conv4(x))
        x = self.bn3(x)
        x = self.dropout3(x)
        
        # Second MaxPooling
        x = self.pool2(x)
        
        # Fifth Convolution Block
        x = F.relu(self.conv5(x))
        
        # Final Convolution Block
        x = self.conv6(x)
        
        # Apply Global Average Pooling
        x = self.gap(x)  # Added GAP operation
        
        # Flatten
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=1)

def get_model():
    return Net()

def save_model(model, accuracy):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mnist_model_{accuracy:.2f}_{timestamp}.pth"
    torch.save(model.state_dict(), f"models/{filename}")
    return filename
