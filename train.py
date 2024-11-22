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


def train_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optim: torch.optim.Optimizer,
    loader: DataLoader,
):
    pbar = tqdm.tqdm(loader)

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

        total_loss += loss_num
        count += 1

        avg_loss = total_loss / count
        pbar.set_description(f"train loss: {avg_loss:.06f}")

    return avg_loss



def val_step(model: torch.nn.Module, loss_fn: torch.nn.Module, loader: DataLoader):
    total_loss = 0.0
    count = 0

    pbar = tqdm.tqdm(loader)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        pred = model(images)
        loss: torch.Tensor = loss_fn(pred, labels)

        assert not torch.isnan(loss)
        loss_num = float(loss.detach().cpu().numpy())

        total_loss += loss_num
        count += 1

        avg_loss = total_loss / count
        pbar.set_description(f"val loss: {avg_loss:.06f}")

    return avg_loss

dataset = CommaDataset("./labeled", chunk_size=8, overlap_size=0, transform=transform)
# data = DummyCommaDataset(length=613, chunk_size=8)
generator = torch.Generator().manual_seed(42)

train_set, val_set = torch.utils.data.random_split(
    dataset, [0.8, 0.2], generator=generator
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=True)

epochs = 100

prev_val_loss = 9999.0

for epoch in range(epochs):
    print(f"epoch: {epoch}")

    train_loss = train_step(model, loss_fn, optim, train_loader)
    val_loss = val_step(model, loss_fn, val_loader)

    if val_loss < prev_val_loss:
        print("saving checkpoint")
        torch.save(model, 


