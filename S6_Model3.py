from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Conv1 block - Input: 28x28x1, Output: 28x28x32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        
        # Conv2 block - Input: 28x28x32, Output: 28x28x32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # First MaxPool - Input: 28x28x32, Output: 14x14x32
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 1x1 Conv1 - Input: 14x14x32, Output: 14x14x16
        self.conv1x1_1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1)
        
        # Conv3 - Input: 14x14x16, Output: 14x14x16
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        
        # Conv4 - Input: 14x14x16, Output: 14x14x16
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        
        # Second MaxPool - Input: 14x14x16, Output: 7x7x16
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 1x1 Conv2 - Input: 7x7x16, Output: 7x7x8
        self.conv1x1_2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1)
        
        # Conv5 - Input: 7x7x8, Output: 7x7x8
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(8)
        self.relu5 = nn.ReLU()
        
        # Conv6 - Input: 7x7x8, Output: 7x7x8
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(8)
        self.relu6 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.1)
        
        # Final 1x1 to reduce channels to 10
        self.final_conv = nn.Conv2d(8, 10, kernel_size=1)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Conv1 and Conv2 blocks
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.mp1(x)
        
        # 1x1 Conv1
        x = self.conv1x1_1(x)
        
        # Conv3 and Conv4 blocks
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.dropout2(x)
        x = self.mp2(x)
        
        # 1x1 Conv2
        x = self.conv1x1_2(x)
        
        # Conv5 and Conv6 blocks
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.dropout3(x)
        
        # Final 1x1 conv
        x = self.final_conv(x)
        
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

