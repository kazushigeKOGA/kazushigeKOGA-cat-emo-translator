import streamlit as st
import numpy as np
import librosa
import onnxruntime as ort

# ONNXモデル読み込み
session = ort.InferenceSession("model_audio_emotion.onnx")

EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]

def predict_emotion(path):
    # 音声読み込み
    y, sr = librosa.load(path, sr=None)
    
    # メルスペクトログラム
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # ONNX入力形式に変換
    # ★ librosa.fix_length は Cloud で壊れるので NumPy で強制リサイズ
    mel_db = np.resize(mel_db, (40, 40))

    mel_db = mel_db.reshape(1, 1, 40, 40).astype(np.float32)

    # 推論
    inputs = {session.get_inputs()[0].name: mel_db}
    outputs = session.run(None, inputs)
    pred = np.argmax(outputs[0])

    return EMOTIONS[pred]

st.title("🎤 音声感情認識（ONNX版 / Streamlit Cloud対応）")

uploaded_file = st.file_uploader("音声ファイルをアップロードしてください（wav）", type=["wav"])

if uploaded_file is not None:
    with open("uploaded.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file)

    emotion = predict_emotion("uploaded.wav")
    st.success(f"判定された感情: **{emotion}**")
