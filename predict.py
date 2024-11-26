import os
import sys
from collections import deque

import cv2
import torch
from torchvision import transforms
from dataloader import load_clip
from model import CalibrationModel, ModelConfig

if not len(sys.argv) > 1:
    raise RuntimeError("No test video provided")

video_path = sys.argv[1]
assert os.path.exists(video_path)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((120, 90)),
    ]
)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")

print(f"device: {device}", file=sys.stderr)

checkpoint = torch.load("model_best.pth", weights_only=True)

model: CalibrationModel = CalibrationModel(config=ModelConfig)
model.to(device)

if torch.mps.is_available():
    model = torch.compile(model, backend="aot_eager")
else:
    model = torch.compile(model)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

context_window = deque(maxlen=8)  # context size of 8 frames

for frame in load_clip(video_path, 0, None):
    frame = transform(frame)
    frame = frame.unsqueeze(0).to(device)

    with torch.no_grad():
        frame_embed = model.forward_backbone(frame)[0]

    context_window.append(frame_embed)

    context_window_tensor = torch.stack(list(context_window))
    context_window_tensor = context_window_tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = model.forward_transformer(context_window_tensor)

    prediction = prediction.squeeze(0)[-1].cpu().numpy()
    print(f"{prediction[0]} {prediction[1]}")
