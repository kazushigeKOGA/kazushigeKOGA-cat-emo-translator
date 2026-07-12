import torch
from audio_model import AudioEmotionCNN

model = AudioEmotionCNN(num_classes=7)
model.load_state_dict(torch.load("model_audio_emotion.pth", map_location="cpu"))
model.eval()

dummy = torch.randn(1, 1, 40, 40)

torch.onnx.export(
    model,
    dummy,
    "model_audio_emotion.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("ONNX model saved as model_audio_emotion.onnx")