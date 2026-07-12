import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # MFCC入力: (batch, 1, 40, 時間フレーム)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 10 * 10, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (1→8)
        x = self.pool(F.relu(self.conv2(x)))  # (8→16)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
