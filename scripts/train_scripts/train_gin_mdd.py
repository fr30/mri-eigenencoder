import hydra
import time
import torch
import torch.nn.functional as F
import wandb

from src.model import GINEncoder, GINClassifier
from src.dataset import RESTfMRIDataset
from src.utils import CosDelayWithWarmupScheduler, IdentityScheduler
from torch_geometric.loader import DataLoader

# from torch.utils.data import DataLoader


def train_epoch(data_loader, model, optimizer, scheduler, device):
    model.train()
    running_loss = 0
    correct = 0

    for data in data_loader:
        data = data.to(device)
        scheduler.adjust_lr(optimizer)
        optimizer.zero_grad()
        out = model(data)
        loss = F.binary_cross_entropy_with_logits(out, data.y.to(torch.float32))
        loss.backward()
        optimizer.step()
        correct += ((out > 0) == data.y).sum().item()
        running_loss += loss.item()

    cum_loss = running_loss * data_loader.batch_size / len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    return cum_loss, acc


@torch.no_grad()
def test_epoch(data_loader, model, device):
    model.eval()
    running_loss = 0
    correct = 0

    for data in data_loader:
        data = data.to(device)
        out = model(data)
        loss = F.binary_cross_entropy_with_logits(out, data.y.to(torch.float32))
        running_loss += loss.item()
        correct += ((out > 0) == data.y).sum().item()

    cum_loss = running_loss * data_loader.batch_size / len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    return cum_loss, acc


@hydra.main(version_base=None, config_path="configs", config_name="gin_mdd")
def main(cfg):
    if cfg.wandb.enabled:
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config={
                "meta": cfg.meta,
                "train": cfg.train,
            },
        )

    if cfg.meta.seed is not None:
        torch.manual_seed(cfg.meta.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_dataset = RESTfMRIDataset(split="train")
    val_dataset = RESTfMRIDataset(split="dev")
    test_dataset = RESTfMRIDataset(split="test")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
    )

    encoder = GINEncoder(
        in_channels=train_dataset.num_nodes,
        hidden_channels=cfg.encoder.hidden_channels,
        num_layers=cfg.encoder.num_layers,
        dropout=cfg.encoder.dropout,
        norm=cfg.encoder.norm,
        emb_style=cfg.encoder.emb_style,
        num_nodes=train_dataset.num_nodes,
        emb_dim=cfg.encoder.emb_dim,
    ).to(device)

    if cfg.encoder.checkpoint_path is not None:
        encoder.load_state_dict(
            torch.load(cfg.encoder.checkpoint_path, weights_only=True)
        )

    model = GINClassifier(
        encoder=encoder,
        num_classes=1,
        linear=cfg.classifier.linear,
    ).to(device)

    if cfg.encoder.freeze:
        for param in model.encoder.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    if cfg.train.lr_scheduler:
        scheduler = CosDelayWithWarmupScheduler(
            cfg.train.lr, len(train_dataloader), cfg.train.epochs
        )
    else:
        scheduler = IdentityScheduler()

    max_acc = 0
    for epoch in range(1, cfg.train.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_epoch(
            train_dataloader, model, optimizer, scheduler, device
        )
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
                    "max_acc": max_acc,
                }
            )
        max_acc = max(max_acc, val_acc)
        print(
            f"Epoch: {epoch}\nTrain loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}, Max acc: {max_acc:.4f}"
        )

    if cfg.meta.test:
        _, test_acc = test_epoch(test_dataloader, model, device)
        if cfg.wandb.enabled:
            run.log({"test_acc": test_acc})
        print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
