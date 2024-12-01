import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First Convolution Block
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)  # input: 28x28x1, output: 26x26x16
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.1)
        
        # Second Convolution Block
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)  # output: 24x24x16
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(0.1)
        
        # First MaxPooling
        self.pool1 = nn.MaxPool2d(2, 2)  # output: 12x12x16
        
        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)  # output: 10x10x32
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout3 = nn.Dropout(0.1)
        
        # Second MaxPooling
        self.pool2 = nn.MaxPool2d(2, 2)  # output: 5x5x32
        
        # Added 1x1 Convolution for channel reduction
        self.conv4 = nn.Conv2d(32, 16, kernel_size=1)  # output: 5x5x16
        self.bn4 = nn.BatchNorm2d(16)
        self.dropout4 = nn.Dropout(0.1)
        
        # Fifth Convolution Block (1x1)
        self.conv5 = nn.Conv2d(16, 10, kernel_size=1)  # output: 5x5x10
        
        # Sixth Convolution Block (5x5)
        self.conv6 = nn.Conv2d(10, 10, kernel_size=5)  # output: 1x1x10
        self.bn5 = nn.BatchNorm2d(10)

    def forward(self, x):
        # First Convolution Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second Convolution Block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # First MaxPooling
        x = self.pool1(x)
        
        # Third Convolution Block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Second MaxPooling
        x = self.pool2(x)
        
        # Added 1x1 Convolution Block
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        
        # Fifth Convolution Block (1x1)
        x = self.conv5(x)
        
        # Sixth Convolution Block (5x5)
        x = self.conv6(x)
        x = self.bn5(x)
        x = F.relu(x)
        
        # Flatten
        x = x.view(-1, 10)
        
        # Final Softmax
        return F.log_softmax(x, dim=1)

def get_model():
    return Net()

def save_model(model, accuracy):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mnist_model_{accuracy:.2f}_{timestamp}.pth"
    torch.save(model.state_dict(), f"models/{filename}")
    return filename