import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from EmotionCNN_light import EmotionCNN_Light

# ============================
# 1. データ前処理
# ============================
transform = transforms.Compose([
    transforms.Resize((64, 64)),   # 軽量モデル用に画像サイズを縮小
    transforms.ToTensor()
])

# ============================
# 2. データセットの読み込み
# ============================
# 例：data/train に感情別フォルダがある前提
train_dataset = datasets.ImageFolder("data/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ============================
# 3. モデルの準備
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN_Light().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============================
# 4. 学習ループ
# ============================
epochs = 10
print("Training started...")

for epoch in range(epochs):
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}")

print("Training finished.")

# ============================
# 5. モデル保存
# ============================
torch.save(model.state_dict(), "model_emotion_light.pth")
print("Saved: model_emotion_light.pth")
