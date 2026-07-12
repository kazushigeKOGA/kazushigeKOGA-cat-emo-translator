import streamlit as st
import torch
import librosa
import numpy as np
from audio_model import AudioEmotionCNN

# フォルダ名の順番に合わせてラベルを設定
labels = ["affection", "angry", "hungry", "surprised", "happy", "relaxed", "sad"]

# モデル読み込み
model = AudioEmotionCNN(num_classes=7)
model.load_state_dict(torch.load("model_audio_emotion.pth", map_location="cpu"))
model.eval()

st.title("Cat Emotion Translator (Audio Version)")

uploaded_file = st.file_uploader("猫の音声ファイルをアップロードしてください", type=["wav", "mp3", "m4a"])

def extract_mfcc(file):
    y, sr = librosa.load(file, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = librosa.util.fix_length(mfcc, size=40, axis=1)
    mfcc = mfcc[:40, :40]
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return mfcc

if uploaded_file is not None:
    st.audio(uploaded_file)

    mfcc = extract_mfcc(uploaded_file)
    with torch.no_grad():
        output = model(mfcc)
        pred = torch.argmax(output, dim=1).item()

    st.write("### 推定された感情：", labels[pred])
