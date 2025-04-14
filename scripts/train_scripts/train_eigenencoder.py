import hydra
import time
import torch
import os
import wandb

from src.model import GINEncoderWithProjector, SFCNEncoderWithProjector
from src.dataset import RestJointDataset, DataLoader
from src.eigenencoder import MCALoss, fmcat_loss
from src.utils import CosDelayWithWarmupScheduler, IdentityScheduler
from tqdm import tqdm


def train_epoch(
    data_loader,
    criterion,
    fmri_enc,
    smri_enc,
    optimizer_f,
    optimizer_s,
    scheduler,
    device,
):
    fmri_enc.train()
    smri_enc.train()
    running_loss = 0
    running_tsd = 0
    smri_fnorm_cum = 0
    fmri_fnorm_cum = 0

    for smri_d, fmri_d, _ in tqdm(data_loader):
        smri_d = smri_d.to(device)
        fmri_d = fmri_d.to(device)

        scheduler.adjust_lr(optimizer_s)
        scheduler.adjust_lr(optimizer_f)

        fmri_f = fmri_enc(fmri_d)
        smri_f = smri_enc(smri_d)

        optimizer_s.zero_grad()
        optimizer_f.zero_grad()
        loss, tsd = criterion(fmri_f, smri_f)
        loss.backward()
        optimizer_s.step()
        optimizer_f.step()

        running_loss += loss.item()
        running_tsd += tsd.item()

        fmri_fnorm_cum += torch.sum(torch.norm(fmri_f, dim=1)).item()
        smri_fnorm_cum += torch.sum(torch.norm(smri_f, dim=1)).item()

    cum_loss = running_loss * data_loader.batch_size / len(data_loader.dataset)
    cum_tsd = running_tsd * data_loader.batch_size / len(data_loader.dataset)
    fmri_fnorm_avg = fmri_fnorm_cum / len(data_loader.dataset)
    smri_fnorm_avg = smri_fnorm_cum / len(data_loader.dataset)

    return cum_loss, cum_tsd, fmri_fnorm_avg, smri_fnorm_avg


@torch.no_grad()
def test_epoch(data_loader, criterion, fmri_enc, smri_enc, device):
    fmri_enc.eval()
    smri_enc.eval()
    running_loss = 0
    running_tsd = 0

    for smri_d, fmri_d, _ in data_loader:
        smri_d = smri_d.to(device)
        fmri_d = fmri_d.to(device)

        fmri_f = fmri_enc(fmri_d)
        smri_f = smri_enc(smri_d)
        loss, tsd = criterion(fmri_f, smri_f)

        running_loss += loss.item()
        running_tsd += tsd.item()

    cum_loss = running_loss * data_loader.batch_size / len(data_loader.dataset)
    cum_tsd = running_tsd * data_loader.batch_size / len(data_loader.dataset)

    return cum_loss, cum_tsd


@hydra.main(version_base=None, config_path="configs", config_name="eigenencoder_ssl")
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

    train_dataset = RestJointDataset(
        split="train", cache_path=cfg.meta.cache_path, inmemory=False
    )
    val_dataset = RestJointDataset(
        split="dev", cache_path=cfg.meta.cache_path, inmemory=False
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.meta.num_workers,
        drop_last=True,
        augment=cfg.train.augment_smri,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.meta.num_workers,
        drop_last=True,
        augment=False,
    )

    smri_enc = SFCNEncoderWithProjector(
        channel_number=cfg.sfcn_encoder.channel_number,
        enc_norm_out=cfg.sfcn_encoder.norm_out,
        emb_dim=cfg.model.emb_dim,
        norm_out=cfg.model.norm_out,
    ).to(device)
    fmri_enc = GINEncoderWithProjector(
        in_channels=train_dataset.fmri.num_nodes,
        hidden_channels=cfg.gin_encoder.hidden_channels,
        num_layers=cfg.gin_encoder.num_layers,
        dropout=cfg.gin_encoder.dropout,
        norm=cfg.gin_encoder.norm,
        emb_style=cfg.gin_encoder.emb_style,
        enc_norm_out=cfg.gin_encoder.norm_out,
        num_nodes=train_dataset.fmri.num_nodes,
        emb_dim=cfg.model.emb_dim,
        norm_out=cfg.model.norm_out,
    ).to(device)

    optimizer_f = torch.optim.Adam(fmri_enc.parameters(), lr=cfg.train.lr)
    optimizer_s = torch.optim.Adam(smri_enc.parameters(), lr=cfg.train.lr)

    criterion = MCALoss(emb_size=cfg.model.emb_dim * 4, device=device)
    # criterion = fmcat_loss

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
        train_loss, train_tsd, fmri_fnorm, smri_fnorm = train_epoch(
            train_dataloader,
            criterion,
            fmri_enc,
            smri_enc,
            optimizer_f,
            optimizer_s,
            scheduler,
            device,
        )
        epoch_time = time.time() - start
        val_loss, val_tsd = test_epoch(
            val_dataloader, criterion, fmri_enc, smri_enc, device
        )
        if cfg.wandb.enabled:
            run.log(
                {
                    "Epoch": epoch,
                    "Train loss": train_loss,
                    "Train TSD": train_tsd,
                    "Validation loss": val_loss,
                    "Val TSD": val_tsd,
                    "Epoch time": epoch_time,
                    "Avg. fMRI f-norm": fmri_fnorm,
                    "Avg. sMRI f-norm": smri_fnorm,
                }
            )
        print(
            f"Epoch: {epoch}\nTrain loss: {train_loss:.4f} tsd: {train_tsd:.4f}, Val loss: {val_loss:.4f} tsd: {val_tsd:.4f}"
        )
        print(f"Avg. fMRI f-norm {fmri_fnorm:.4f}, Avg. sMRI f-norm: {smri_fnorm:.4f}")

        if cfg.meta.save_model_freq > 0 and epoch % cfg.meta.save_model_freq == 0:
            torch.save(
                fmri_enc.encoder.state_dict(),
                f"{cfg.meta.save_model_path}/fmri_enc_ep{epoch}.pt",
            )
            torch.save(
                smri_enc.encoder.state_dict(),
                f"{cfg.meta.save_model_path}/smri_enc_ep{epoch}.pt",
            )


if __name__ == "__main__":
    main()
