import hydra
import time
import torch
import torch.nn.functional as F
import wandb

from tqdm import tqdm
from src.model import SFCNClassifier
from src.dataset import RESTsMRIDataset
from torch.utils.data import DataLoader


def train_epoch(data_loader, model, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0

    for x, y in tqdm(data_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.binary_cross_entropy_with_logits(out, y.to(torch.float32))
        loss.backward()
        optimizer.step()
        correct += ((out > 0) == y).sum().item()
        running_loss += loss.item()

    cum_loss = running_loss * data_loader.batch_size / len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    return cum_loss, acc


@torch.no_grad()
def test_epoch(data_loader, model, device):
    model.eval()
    running_loss = 0
    correct = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = F.binary_cross_entropy_with_logits(out, y.to(torch.float32))
        running_loss += loss.item()
        correct += ((out > 0) == y).sum().item()

    cum_loss = running_loss * data_loader.batch_size / len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    return cum_loss, acc


@hydra.main(version_base=None, config_path="configs", config_name="sfcn_mdd")
def main(cfg):
    if cfg.wandb.enabled:
        run = wandb.init(
            # entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            config={
                "meta": cfg.meta,
                "train": cfg.train,
                "model": cfg.model,
            },
        )

    if cfg.meta.seed is not None:
        torch.manual_seed(cfg.meta.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_dataset = RESTsMRIDataset(split="train")
    val_dataset = RESTsMRIDataset(split="dev")
    test_dataset = RESTsMRIDataset(split="test")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.meta.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.meta.num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.meta.num_workers,
    )

    model = SFCNClassifier(channel_number=cfg.model.channel_number, output_dim=1).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    if cfg.train.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.train.lr_scheduler_step_size, gamma=0.5
        )

    max_acc = 0
    for epoch in range(1, cfg.train.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_epoch(train_dataloader, model, optimizer, device)
        epoch_time = time.time() - start
        val_loss, val_acc = test_epoch(val_dataloader, model, device)
        if cfg.wandb.enabled:
            run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch_time": epoch_time,
                }
            )
        max_acc = max(max_acc, val_acc)
        print(
            f"Epoch: {epoch}\nTrain loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}, Max acc: {max_acc:.4f}"
        )
        print(f"Epoch time: {epoch_time:.2f}s")

        if cfg.train.lr_scheduler:
            scheduler.step()

    if cfg.meta.test:
        _, test_acc = test_epoch(test_dataloader, model, device)
        if cfg.wandb.enabled:
            run.log({"test_acc": test_acc})
        print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
