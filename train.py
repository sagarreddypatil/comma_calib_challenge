import tqdm
from dataloader import CommaDataset, DummyCommaDataset
from model import CalibrationModel, ModelConfig
import torch
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import os
import random
import numpy as np


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Save checkpoint to disk."""
    print(f"Saving checkpoint to {filename}")
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer):
    """Load checkpoint from disk."""
    if not os.path.isfile(filename):
        return 0, float("inf")  # Start from epoch 0 if no checkpoint

    print(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint["best_val_loss"]


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

        means, logvars = model(images)
        loss = model.loss(means, logvars, labels)

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

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            pred, _ = model(images)
            loss: torch.Tensor = loss_fn(pred, labels)

            assert not torch.isnan(loss)
            loss_num = float(loss.detach().cpu().numpy())

            total_loss += loss_num
            count += 1

            avg_loss = total_loss / count
            pbar.set_description(f"val loss: {avg_loss:.06f}")

    return avg_loss


# Set random seed for reproducibility
set_seed(42)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")

print(f"device: {device}")

base_model = CalibrationModel(config=ModelConfig).to(device)
optim = Adam(base_model.parameters(), lr=1e-4)

# Compile model if supported
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

# Use a fixed generator for reproducible train/val split
generator = torch.Generator().manual_seed(42)
train_set, val_set = torch.utils.data.random_split(
    dataset, [0.5, 0.5], generator=generator
)

train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True,
    worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id),
)
val_loader = DataLoader(
    val_set, batch_size=64, shuffle=False
)  # No need to shuffle validation set

epochs = 100
checkpoint_path = "model_checkpoint.pth.tar"
best_checkpoint_path = "model_best.pth.tar"

# Load checkpoint if exists
start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optim)
writer = SummaryWriter()

for epoch in range(start_epoch, start_epoch + epochs):
    print(f"epoch: {epoch}")

    train_loss = train_step(model, loss_fn, optim, train_loader)
    val_loss = val_step(model, loss_fn, val_loader)

    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("val_loss", val_loss, epoch)

    # Save regular checkpoint
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }
    save_checkpoint(checkpoint, checkpoint_path)

    # Save best model checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(checkpoint, best_checkpoint_path)
        print(f"New best validation loss: {best_val_loss:.6f}")
