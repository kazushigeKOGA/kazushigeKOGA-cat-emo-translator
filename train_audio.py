import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from audio_model import AudioEmotionCNN

# ============================
# 1. データセット
# ============================
class AudioDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = []
        self.labels = []
        self.classes = sorted(os.listdir(root))

        for idx, cls in enumerate(self.classes):
            folder = os.path.join(root, cls)
            for f in os.listdir(folder):
                if f.endswith((".wav", ".mp3", ".m4a", ".m4p")):
                    self.files.append(os.path.join(folder, f))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]

        y, sr = librosa.load(path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        mfcc = librosa.util.fix_length(mfcc, size=40, axis=1)
        mfcc = mfcc[:40, :40]

        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
        return mfcc, label

# ============================
# 2. 学習設定
# ============================
dataset = AudioDataset("audio_data/train")
print("Training started. Total files:", len(dataset))
loader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioEmotionCNN(num_classes=7).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ============================
# 3. 学習ループ
# ============================
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for mfcc, label in loader:
        mfcc, label = mfcc.to(device), label.to(device)

        optimizer.zero_grad()
        out = model(mfcc)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}")

# ============================
# 4. 保存
# ============================
torch.save(model.state_dict(), "model_audio_emotion.pth")
print("Saved model_audio_emotion.pth")
