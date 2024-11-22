from dataloader import CommaDataset, DummyCommaDataset
from model import CalibrationModel, ModelConfig

import tqdm

import torch
from torch.optim.adam import Adam

from torch.utils.data import DataLoader
from torchvision import transforms

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

print(f"device: {device}")

model = CalibrationModel(config=ModelConfig)

if torch.mps.is_available():
    model = torch.compile(model, backend="aot_eager")
else:
    model = torch.compile(model)
model.to(device)

optim = Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

data = CommaDataset("./labeled", chunk_size=8, overlap_size=0, transform=transform)
# data = DummyCommaDataset(length=613, chunk_size=8)
train_loader = DataLoader(data, batch_size=64, shuffle=True)

epochs = 50

for epoch in range(epochs):
    pbar = tqdm.tqdm(train_loader)

    total_loss = 0
    count = 0

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        pred = model(images)
        loss: torch.Tensor = loss_fn(pred, labels)

        assert not torch.isnan(loss)

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_num = float(loss.detach().cpu().numpy())
        pbar.set_description(f"loss: {loss_num:.06f}")

        total_loss += loss_num
        count += 1

    avg_loss = total_loss / count
    pbar.set_description(f"loss: {avg_loss:.06f}")
