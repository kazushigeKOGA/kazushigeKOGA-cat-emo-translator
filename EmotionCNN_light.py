import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN_Light(nn.Module):
    def __init__(self):
        super().__init__()

        # --- 軽量化ポイント ---
        # ① チャンネル数を大幅に削減（16→8、32→16）
        # ② 畳み込み層を2層に減らす（元は3層）
        # ③ 全結合層を小さくする（128→64）
        # ④ 入力画像サイズを 64×64 前提に最適化

        # Conv1: 3ch → 8ch
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)

        # Conv2: 8ch → 16ch
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

        # MaxPooling（2×2）
        self.pool = nn.MaxPool2d(2, 2)

        # 64×64 → conv1 → pool → 32×32  
        # → conv2 → pool → 16×16  
        # → 16ch × 16 × 16 = 4096
        self.fc1 = nn.Linear(16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 5)  # 感情分類5種類想定

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 64→32
        x = self.pool(F.relu(self.conv2(x)))  # 32→16
        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
