import hydra
import time
import torch
import os
import wandb
import warnings

from accelerate import Accelerator
from src.model import BarlowTwinsGIN
from src.dataset import RESTfMRIDataset, BTDataLoader
from src.loss import BTLoss
from src.optim import LARS
from src.utils import CosDelayWithWarmupScheduler, IdentityScheduler
from tqdm import tqdm


# PyGCL produces too many deprecation warnings, so we ignore them
warnings.filterwarnings("ignore", category=UserWarning)


def train_epoch(
    data_loader,
    criterion,
    model,
    optimizer,
    scheduler,
    device,
):
    model.train()
    running_loss = 0

    for x1, x2 in tqdm(data_loader):
        x1 = x1.to(device)
        x2 = x2.to(device)

        scheduler.adjust_lr(optimizer)
        y1, y2 = model(x1, x2)

        optimizer.zero_grad()

        loss = criterion(y1, y2)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    cum_loss = running_loss * data_loader.batch_size / len(data_loader.dataset)

    return cum_loss


@torch.no_grad()
def test_epoch(data_loader, criterion, model, device):
    model.eval()
    running_loss = 0

    for x1, x2 in data_loader:
        x1 = x1.to(device)
        x2 = x2.to(device)

        y1, y2 = model(x1, x2)
        loss = criterion(y1, y2)
        running_loss += loss.item()

    cum_loss = running_loss * data_loader.batch_size / len(data_loader.dataset)

    return cum_loss


@hydra.main(version_base=None, config_path="configs", config_name="barlow_twins_graph")
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

    train_dataset = RESTfMRIDataset(split="train", cache_path=cfg.meta.cache_path)
    val_dataset = RESTfMRIDataset(split="dev", cache_path=cfg.meta.cache_path)

    train_dataloader = BTDataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.meta.num_workers,
        drop_last=False,
    )

    val_dataloader = BTDataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.meta.num_workers,
        drop_last=False,
    )

    model = BarlowTwinsGIN(
        in_channels=train_dataset.num_nodes,
        hidden_channels=cfg.gin_encoder.hidden_channels,
        num_layers=cfg.gin_encoder.num_layers,
        dropout=cfg.gin_encoder.dropout,
        norm=cfg.gin_encoder.norm,
        emb_style=cfg.gin_encoder.emb_style,
        num_nodes=train_dataset.num_nodes,
        emb_dim=cfg.model.emb_dim,
        norm_out=cfg.model.norm_out,
    ).to(device)

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

    criterion = BTLoss(batch_size=cfg.train.batch_size, lambd=cfg.train.lambd)

    if cfg.train.lr_scheduler:
        scheduler = CosDelayWithWarmupScheduler(
            cfg.train.lr, len(train_dataloader), cfg.train.epochs
        )
    else:
        scheduler = IdentityScheduler()

    if cfg.meta.save_model_freq > 0:
        if not os.path.exists(cfg.meta.save_model_path):
            os.makedirs(cfg.meta.save_model_path)

    for epoch in range(1, cfg.train.epochs + 1):
        start = time.time()
        train_loss = train_epoch(
            train_dataloader,
            criterion,
            model,
            optimizer,
            scheduler,
            device,
        )
        epoch_time = time.time() - start
        val_loss = test_epoch(val_dataloader, criterion, model, device)
        if cfg.wandb.enabled:
            run.log(
                {
                    "Train loss": train_loss,
                    "Validation loss": val_loss,
                }
            )
        print(
            f"Epoch: {epoch}\nTrain loss: {train_loss:.4f} Val loss: {val_loss:.4f}, Time: {epoch_time:.2f}s"
        )

        if cfg.meta.save_model_freq > 0 and epoch % cfg.meta.save_model_freq == 0:
            torch.save(
                model.encoder.state_dict(),
                f"{cfg.meta.save_model_path}/fmri_enc_ep{epoch}.pt",
            )


if __name__ == "__main__":
    main()
