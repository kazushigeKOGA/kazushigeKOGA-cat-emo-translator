import streamlit as st
import numpy as np
import librosa
import onnxruntime as ort
from pydub import AudioSegment
import os

session = ort.InferenceSession("model_audio_emotion.onnx")

EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]

def convert_to_wav(input_path, output_path="converted.wav"):
    """
    Cloud で最も安定する m4a → wav 変換
    ffmpeg ではなく pydub を使う
    """
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        st.error(f"音声変換に失敗しました: {e}")
        return None

def predict_emotion(path):
    y, sr = librosa.load(path, sr=None)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = librosa.util.fix_length(mel_db, size=40, axis=1)
    mel_db = mel_db.reshape(1, 1, 40, 40).astype(np.float32)

    inputs = {session.get_inputs()[0].name: mel_db}
    outputs = session.run(None, inputs)
    pred = np.argmax(outputs[0])

    return EMOTIONS[pred]

st.title("🎤 音声感情認識（ONNX版 / Streamlit Cloud対応）")

uploaded_file = st.file_uploader("音声ファイルをアップロードしてください（wav / m4a / mp3）",
                                 type=["wav", "m4a", "mp3"])

if uploaded_file is not None:
    raw_path = "uploaded.raw"
    with open(raw_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    ext = uploaded_file.name.split(".")[-1].lower()

    if ext == "wav":
        wav_path = raw_path
    else:
        wav_path = convert_to_wav(raw_path)

    if wav_path:
        st.audio(uploaded_file)
        emotion = predict_emotion(wav_path)
        st.success(f"判定された感情: **{emotion}**")

    if os.path.exists(raw_path):
        os.remove(raw_path)
    if os.path.exists("converted.wav"):
        os.remove("converted.wav")
