import streamlit as st
import numpy as np
import onnxruntime as ort
import wave
import os
from scipy.signal import stft

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

def mel_filterbank(sr, n_fft=512, n_mels=40):
    # メルフィルタバンクを自前で作成
    fmin, fmax = 0, sr / 2
    mel_min = 2595 * np.log10(1 + fmin / 700)
    mel_max = 2595 * np.log10(1 + fmax / 700)

    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)

    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_left = bins[m - 1]
        f_center = bins[m]
        f_right = bins[m + 1]

        for k in range(f_left, f_center):
            fb[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            fb[m - 1, k] = (f_right - k) / (f_right - f_center)

    return fb

def compute_mel(y, sr):
    # STFT
    f, t, Zxx = stft(y, sr, nperseg=512)
    S = np.abs(Zxx)

    # メルフィルタバンク
    fb = mel_filterbank(sr, n_fft=512, n_mels=40)

    mel = np.dot(fb, S)

    # 対数
    mel_db = np.log10(mel + 1e-6)

    # 時間方向を40に固定
    mel_db = mel_db[:, :40]
    mel_db = np.pad(mel_db, ((0, 0), (0, max(0, 40 - mel_db.shape[1]))))

    return mel_db

def predict_emotion(path):
    y, sr = load_wav(path)

    mel_db = compute_mel(y, sr)

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
