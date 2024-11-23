import torch
from model import CalibrationModel

model = CalibrationModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()