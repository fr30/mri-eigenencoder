import hydra
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import wandb
import warnings

from sklearn.model_selection import StratifiedKFold
from src.optim import LARS
from src.model import GPSEncoder, Classifier
from src.dataset import RESTfMRIDataset, GPSCachedLoader, GPSConcatDataset
from src.utils import CosDelayWithWarmupScheduler, IdentityScheduler
from torch_geometric.loader import DataLoader


warnings.filterwarnings("ignore", category=UserWarning)


def run_training(cfg, train_dataloader, val_dataloader, model, optimizer, device):
    if cfg.train.lr_scheduler:
        scheduler = CosDelayWithWarmupScheduler(
            cfg.train.lr, len(train_dataloader.dataset), cfg.train.epochs
        )
    else:
        scheduler = IdentityScheduler()

    max_acc = 0
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []

    for epoch in range(1, cfg.train.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_epoch(
            train_dataloader, model, optimizer, scheduler, device
        )
        epoch_time = time.time() - start
        val_loss, val_acc = test_epoch(val_dataloader, model, device)

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        max_acc = max(max_acc, val_acc)
        print(
            f"Epoch: {epoch}\nTrain loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}, Max acc: {max_acc:.4f}"
        )
        print(f"Epoch time: {epoch_time:.2f}s", flush=True)

    return train_accs, train_losses, val_accs, val_losses


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


@hydra.main(version_base=None, config_path="configs", config_name="gps_sex")
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
    else:
        run = None

    if cfg.meta.seed is not None:
        torch.manual_seed(cfg.meta.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rest_dataset = RESTfMRIDataset(split="full", label="sex")
    dataset = GPSConcatDataset([rest_dataset])
    kf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=cfg.meta.seed,
    )

    splits = kf.split(dataset, rest_dataset.labels)
    run_train_accs = []
    run_val_accs = []
    run_train_losses = []
    run_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        train_set = torch.utils.data.dataset.Subset(dataset, train_idx)
        val_set = torch.utils.data.dataset.Subset(dataset, val_idx)

        train_dataloader = DataLoader(
            dataset=train_set,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.meta.num_workers,
            shuffle=True,
            pin_memory=False,
        )
        val_dataloader = DataLoader(
            dataset=val_set,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.meta.num_workers,
            pin_memory=False,
        )

        encoder = GPSEncoder(
            in_channels=dataset.num_nodes,
            emb_dim=cfg.encoder.emb_dim,
            pe_dim=cfg.encoder.pe_dim,
            num_layers=cfg.encoder.num_layers,
            dropout=cfg.encoder.dropout,
            norm_out=cfg.encoder.norm,
            attn_type=cfg.encoder.attn_type,
        ).to(device)

        if cfg.encoder.checkpoint_path is not None:
            encoder.load_state_dict(
                torch.load(cfg.encoder.checkpoint_path, weights_only=True)
            )

        model = Classifier(
            encoder=encoder,
            num_classes=1,
            linear=cfg.classifier.linear,
        ).to(device)

        if cfg.encoder.freeze:
            for param in model.encoder.parameters():
                param.requires_grad = False

        if cfg.train.lars:
            param_weights = []
            param_biases = []
            for param in model.parameters():
                if param.ndim == 1:
                    param_biases.append(param)
                else:
                    param_weights.append(param)
            parameters = [{"params": param_weights}, {"params": param_biases}]
            optimizer = LARS(
                parameters,
                lr=0,
                weight_decay=cfg.train.weight_decay,
                weight_decay_filter=True,
                lars_adaptation_filter=True,
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
            )

        print(f"Fold: {fold + 1}")
        train_accs, train_losses, val_accs, val_losses = run_training(
            cfg=cfg,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        run_train_accs.append(train_accs)
        run_val_accs.append(val_accs)
        run_train_losses.append(train_losses)
        run_val_losses.append(val_losses)

    train_loss_mean = np.mean(run_train_losses, axis=0)
    val_loss_mean = np.mean(run_val_losses, axis=0)
    train_acc_mean = np.mean(run_train_accs, axis=0)
    train_acc_std = np.std(run_train_accs, axis=0)
    val_acc_mean = np.mean(run_val_accs, axis=0)
    val_acc_std = np.std(run_val_accs, axis=0)

    if cfg.wandb.enabled:
        for i in range(len(train_loss_mean)):
            run.log(
                {
                    "train_loss": train_loss_mean[i],
                    "train_acc": train_acc_mean[i],
                    "train_acc_std": train_acc_std[i],
                    "val_loss": val_loss_mean[i],
                    "val_acc": val_acc_mean[i],
                    "val_acc_std": val_acc_std[i],
                }
            )


if __name__ == "__main__":
    main()
