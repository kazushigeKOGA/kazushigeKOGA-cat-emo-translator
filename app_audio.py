import streamlit as st
import numpy as np
import librosa
import onnxruntime as ort
from pydub import AudioSegment
import os

session = ort.InferenceSession("model_audio_emotion.onnx")

EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]

def load_audio(path):
    """
    m4a / mp3 / wav を pydub で直接読み込み numpy に変換
    → librosa.load を使わないので Cloud の soundfile バグを完全回避
    """
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= np.max(np.abs(samples))  # 正規化

    return samples, 16000

def predict_emotion(path):
    # pydub で読み込む（librosa.load は使わない）
    y, sr = load_audio(path)

    # メルスペクトログラム
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 横方向（時間軸）だけ40に固定
    mel_db = librosa.util.fix_length(mel_db, size=40, axis=1)

    mel_db = mel_db.reshape(1, 1, 40, 40).astype(np.float32)

    inputs = {session.get_inputs()[0].name: mel_db}
    outputs = session.run(None, inputs)
    pred = np.argmax(outputs[0])

    return EMOTIONS[pred]

st.title("🎤 音声感情認識（ONNX版 / Streamlit Cloud対応）")

uploaded_file = st.file_uploader(
    "音声ファイルをアップロードしてください（wav / m4a / mp3）",
    type=["wav", "m4a", "mp3"]
)

if uploaded_file is not None:
    raw_path = "uploaded.raw"
    with open(raw_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file)

    emotion = predict_emotion(raw_path)
    st.success(f"判定された感情: **{emotion}**")

    os.remove(raw_path)
