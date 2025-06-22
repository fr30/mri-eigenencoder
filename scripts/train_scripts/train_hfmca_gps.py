import hydra
import time
import torch
import os
import wandb
import warnings

from src.model import HFMCAGPS
from src.dataset import (
    RESTfMRIDataset,
    HFMCADataLoader,
    ABIDEfMRIDataset,
    AOMICfMRIDataset,
    BSNIPfMRIDataset,
    HCPfMRIDataset,
    GPSConcatDataset,
)
from src.loss import HFMCALoss
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

    for x in tqdm(data_loader):
        x = x.to(device)
        scheduler.adjust_lr(optimizer)
        y, y_dash = model(x)

        optimizer.zero_grad()

        loss = criterion(y, y_dash)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    cum_loss = running_loss * data_loader.batch_size / len(data_loader.dataset)

    return cum_loss


@torch.no_grad()
def test_epoch(data_loader, criterion, model, device):
    model.eval()
    running_loss = 0

    for x in data_loader:
        x = x.to(device)
        y, y_dash = model(x)
        loss = criterion(y, y_dash)
        running_loss += loss.item()

    cum_loss = running_loss * data_loader.batch_size / len(data_loader.dataset)

    return cum_loss


@hydra.main(version_base=None, config_path="configs", config_name="hfmca_gps")
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
    aomic_dataset = AOMICfMRIDataset(split="full")
    bsnip_dataset = BSNIPfMRIDataset(split="full")
    hcp_dataset = HCPfMRIDataset(split="train")

    train_dataset = GPSConcatDataset(
        [rest_train_dataset, abide_dataset, aomic_dataset, bsnip_dataset, hcp_dataset]
    )
    val_dataset = GPSConcatDataset([rest_val_dataset])

    train_dataloader = HFMCADataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.meta.num_workers,
        drop_last=True,
        num_views=cfg.train.num_views,
    )

    val_dataloader = HFMCADataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.meta.num_workers,
        drop_last=False,
        num_views=cfg.train.num_views,
    )

    model = HFMCAGPS(
        in_channels=rest_train_dataset.num_nodes,
        emb_dim=cfg.encoder.emb_dim,
        pe_dim=cfg.encoder.pe_dim,
        num_layers=cfg.encoder.num_layers,
        dropout=cfg.encoder.dropout,
        norm_out=cfg.encoder.norm,
        attn_type=cfg.encoder.attn_type,
        nviews=cfg.train.num_views,
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

    criterion = HFMCALoss(device=device)

    if cfg.train.lr_scheduler:
        scheduler = CosDelayWithWarmupScheduler(
            cfg.train.lr, len(train_dataloader), cfg.train.epochs
        )
    else:
        scheduler = IdentityScheduler()

    if cfg.meta.save_model_freq > 0:
        if not os.path.exists(cfg.meta.save_model_path):
            os.makedirs(cfg.meta.save_model_path)

    for epoch in range(0, cfg.train.epochs + 1):
        if cfg.meta.save_model_freq > 0 and epoch % cfg.meta.save_model_freq == 0:
            torch.save(
                model.encoder.state_dict(),
                f"{cfg.meta.save_model_path}/fmri_enc_ep{epoch}.pt",
            )

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


if __name__ == "__main__":
    main()
