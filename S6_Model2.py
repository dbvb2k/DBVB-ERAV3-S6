import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

# Third model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First Block
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)  # input: 28x28x1, output: 26x26x8
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)  # output: 24x24x16
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.1)
        
        # First MaxPool + 1x1 Conv
        self.pool1 = nn.MaxPool2d(2, 2)  # output: 12x12x16
        self.conv3 = nn.Conv2d(16, 8, kernel_size=1)  # output: 12x12x8
        
        # Third Block
        self.bn3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3)  # output: 10x10x16
        self.bn4 = nn.BatchNorm2d(16)
        
        # Fourth Block
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3)  # output: 8x8x16
        self.bn5 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(0.1)
        
        # Second MaxPool + 1x1 Conv
        self.pool2 = nn.MaxPool2d(2, 2)  # output: 4x4x16
        self.conv6 = nn.Conv2d(16, 8, kernel_size=1)  # output: 4x4x8
        self.bn6 = nn.BatchNorm2d(8)
        
        # Fifth Block
        self.conv7 = nn.Conv2d(8, 16, kernel_size=3)  # output: 2x2x16
        self.bn7 = nn.BatchNorm2d(16)
        
        # Final Conv
        self.conv8 = nn.Conv2d(16, 10, kernel_size=2)  # output: 1x1x10
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # First Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second Block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # First MaxPool + 1x1 Conv
        x = self.pool1(x)
        x = self.conv3(x)
        
        # Third Block
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Fourth Block
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Second MaxPool + 1x1 Conv
        x = self.pool2(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        
        # Fifth Block
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        
        # Final Conv
        x = self.conv8(x)
        
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
