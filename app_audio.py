import streamlit as st
import numpy as np
import librosa
import onnxruntime as ort
import wave
import os

session = ort.InferenceSession("model_audio_emotion.onnx")

EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]

def load_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        audio = wf.readframes(n)

    y = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
    y /= np.max(np.abs(y))
    return y, sr

def predict_emotion(path):
    y, sr = load_wav(path)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = librosa.util.fix_length(mel_db, size=40, axis=1)
    mel_db = mel_db.reshape(1, 1, 40, 40).astype(np.float32)

    inputs = {session.get_inputs()[0].name: mel_db}
    outputs = session.run(None, inputs)
    pred = np.argmax(outputs[0])

    return EMOTIONS[pred]

st.title("🎤 音声感情認識（WAV専用 / Streamlit Cloud対応）")

uploaded_file = st.file_uploader(
    "音声ファイルをアップロードしてください（wav のみ対応）",
    type=["wav"]
)

if uploaded_file is not None:
    raw_path = "uploaded.wav"
    with open(raw_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file)

    emotion = predict_emotion(raw_path)
    st.success(f"判定された感情: **{emotion}**")

    os.remove(raw_path)
