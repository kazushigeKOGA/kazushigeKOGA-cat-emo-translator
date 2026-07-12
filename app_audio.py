import streamlit as st
import numpy as np
import librosa
import onnxruntime as ort

session = ort.InferenceSession("model_audio_emotion.onnx")

EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]

def predict_emotion(path):
    y, sr = librosa.load(path, sr=None)

    # メルスペクトログラム（縦40は固定）
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 横方向（時間軸）だけを40に固定する
    # → 特徴量が壊れない
    mel_db = librosa.util.fix_length(mel_db, size=40, axis=1)

    # 縦方向はすでに40なのでそのまま
    mel_db = mel_db.reshape(1, 1, 40, 40).astype(np.float32)

    inputs = {session.get_inputs()[0].name: mel_db}
    outputs = session.run(None, inputs)
    pred = np.argmax(outputs[0])

    return EMOTIONS[pred]

st.title("🎤 音声感情認識（ONNX版 / Streamlit Cloud対応）")

uploaded_file = st.file_uploader("音声ファイルをアップロードしてください（wav）", type=["wav", "m4a"])

if uploaded_file is not None:
    with open("uploaded.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file)

    emotion = predict_emotion("uploaded.wav")
    st.success(f"判定された感情: **{emotion}**")
