import hydra
import time
import torch
import os
import wandb
import warnings

from src.model import BarlowTwinsGPS
from src.dataset import (
    RESTfMRIDataset,
    BTDataLoader,
    ABIDEfMRIDataset,
    AOMICfMRIDataset,
    BSNIPfMRIDataset,
    HCPfMRIDataset,
    GPSConcatDataset,
)
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


@hydra.main(version_base=None, config_path="configs", config_name="barlow_twins_gps")
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

    rest_train_dataset = RESTfMRIDataset(split="train", cache_path=cfg.meta.cache_path)
    rest_val_dataset = RESTfMRIDataset(split="dev")
    abide_dataset = ABIDEfMRIDataset(split="train")
    hcp_dataset = HCPfMRIDataset(split="train")
    bsnip_dataset = BSNIPfMRIDataset(split="train")
    aomic_dataset = AOMICfMRIDataset(split="full")

    train_dataset = GPSConcatDataset(
        [rest_train_dataset, abide_dataset, hcp_dataset, aomic_dataset, bsnip_dataset]
    )
    val_dataset = GPSConcatDataset([rest_val_dataset])

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

    model = BarlowTwinsGPS(
        in_channels=rest_train_dataset.num_nodes,
        emb_dim=cfg.encoder.emb_dim,
        pe_dim=cfg.encoder.pe_dim,
        num_layers=cfg.encoder.num_layers,
        dropout=cfg.encoder.dropout,
        norm_out=cfg.encoder.norm,
        attn_type=cfg.encoder.attn_type,
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
