import tqdm

from dataloader import CommaDataset, DummyCommaDataset
from model import CalibrationModel, ModelConfig

import torch
from torch.optim.adam import Adam

from torch.utils.data import DataLoader
from torchvision import transforms


def train_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optim: torch.optim.Optimizer,
    loader: DataLoader,
):
    model.train()
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
    model.eval()

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


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")

print(f"device: {device}")

base_model = CalibrationModel(config=ModelConfig).to(device)
optim = Adam(base_model.parameters(), lr=1e-4)
start_epoch = 0


if torch.mps.is_available():
    model = torch.compile(base_model, backend="aot_eager")
else:
    model = torch.compile(base_model)

loss_fn = torch.nn.MSELoss()


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((120, 90)),
    ]
)

dataset = CommaDataset("./labeled", chunk_size=8, overlap_size=0, transform=transform)
# data = DummyCommaDataset(length=613, chunk_size=8)

train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=True)

epochs = 100

prev_val_loss = val_step(model, loss_fn, val_loader)

for epoch in range(start_epoch, start_epoch + epochs):
    print(f"epoch: {epoch}")

    train_loss = train_step(model, loss_fn, optim, train_loader)
    val_loss = val_step(model, loss_fn, val_loader)
