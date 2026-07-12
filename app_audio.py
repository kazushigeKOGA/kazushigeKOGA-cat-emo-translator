import streamlit as st
import numpy as np
import librosa
import onnxruntime as ort
import subprocess
import os

session = ort.InferenceSession("model_audio_emotion.onnx")

EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]

def convert_m4a_to_mp3(input_path, output_path="converted.mp3"):
    """Cloud で最も安定する m4a → mp3 変換"""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-acodec", "libmp3lame",
                "-ar", "16000",
                "-ac", "1",
                output_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return output_path
    except Exception as e:
        st.error(f"音声変換に失敗しました: {e}")
        return None

def load_audio(path):
    """mp3 / wav を librosa で読み込む（Cloud で安定）"""
    y, sr = librosa.load(path, sr=None)
    return y, sr

def predict_emotion(path):
    y, sr = load_audio(path)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)

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

    ext = uploaded_file.name.split(".")[-1].lower()

    if ext == "m4a":
        mp3_path = convert_m4a_to_mp3(raw_path)
        wav_path = mp3_path
    else:
        wav_path = raw_path

    if wav_path:
        st.audio(uploaded_file)
        emotion = predict_emotion(wav_path)
        st.success(f"判定された感情: **{emotion}**")

    if os.path.exists(raw_path):
        os.remove(raw_path)
    if os.path.exists("converted.mp3"):
        os.remove("converted.mp3")
