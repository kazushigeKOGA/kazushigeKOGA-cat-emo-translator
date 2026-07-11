import torch
import torch.nn as nn
import torchaudio
import numpy as np
import gradio as gr

# ===== あなたの学習モデル構造 =====
class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 16 * 100, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ===== モデル読み込み =====
model = EmotionCNN()
state_dict = torch.load("model_emotion.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# ===== 音声処理 =====
def preprocess_audio(audio):
    sr, data = audio

    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)

    if data.dim() > 1:
        data = torch.mean(data, dim=1)

    if sr != 16000:
        data = torchaudio.functional.resample(data, sr, 16000)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )(data)

    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

    T = mel_db.shape[-1]
    FIX_LEN = 400
    if T < FIX_LEN:
        mel_db = nn.functional.pad(mel_db, (0, FIX_LEN - T))
    else:
        mel_db = mel_db[:, :FIX_LEN]

    mel_db = mel_db.unsqueeze(0)
    return mel_db

# ===== 推論 =====
def predict_emotion(audio):
    try:
        mel = preprocess_audio(audio)
        with torch.no_grad():
            output = model(mel)
            pred = torch.argmax(output, dim=1).item()

        emotions = ["affection", "hungry", "angry", "play"]
        return f"猫の感情: {emotions[pred]}"

    except Exception as e:
        return f"エラー: {str(e)}"

# ===== Gradio UI =====
app = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(type="numpy", label="猫の音声をアップロード（m4a / wav）"),
    outputs=gr.Textbox(label="判定結果"),
    title="Cat Emo Translator",
    description="猫の鳴き声から感情を推定します（ZeroGPU対応）"
)

app.launch()

