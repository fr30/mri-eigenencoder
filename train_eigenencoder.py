import hydra
import time
import torch
import torch.nn.functional as F
import wandb

from src.model import GINEncoder, SFCNEncoder
from src.dataset import RestJointDataset, DataLoader
from src.eigenencoder import MCALoss, fmcat_loss
from tqdm import tqdm


def train_epoch(
    data_loader, criterion, fmri_enc, smri_enc, optimizer_f, optimizer_s, device
):
    fmri_enc.train()
    smri_enc.train()
    running_loss = 0
    running_tsd = 0
    rfnorm_cum = 0
    ffnorm_cum = 0
    fgradnorm_cum = 0
    step = 0

    # for smri_d, fmri_d, _ in tqdm(data_loader):
    for smri_d, fmri_d, _ in data_loader:
        smri_d = smri_d.to(device)
        fmri_d = fmri_d.to(device)

        optimizer_s.zero_grad()
        optimizer_f.zero_grad()

        fmri_f = fmri_enc(fmri_d)
        smri_f = smri_enc(smri_d)

        if step == 0:
            print("Norms")
            print(fmri_f.norm().item(), smri_f.norm().item())

        step += 1

        loss, tsd = criterion(fmri_f, smri_f)

        rfnorm_cum += torch.sum(torch.norm(fmri_f, 1)).item()
        ffnorm_cum += torch.sum(torch.norm(smri_f, 1)).item()

        loss.backward()
        optimizer_s.step()
        optimizer_f.step()

        running_loss += loss.item()
        running_tsd += tsd.item()

    cum_loss = running_loss * data_loader.batch_size / len(data_loader.dataset)
    cum_tsd = running_tsd * data_loader.batch_size / len(data_loader.dataset)
    rfnorm_avg = rfnorm_cum / len(data_loader.dataset)
    ffnorm_avg = ffnorm_cum / len(data_loader.dataset)
    fgradnorm_avg = fgradnorm_cum / len(data_loader.dataset)

    return cum_loss, cum_tsd, rfnorm_avg, ffnorm_avg, fgradnorm_avg


@torch.no_grad()
def test_epoch(data_loader, criterion, fmri_enc, smri_enc, device):
    fmri_enc.eval()
    smri_enc.eval()
    running_loss = 0
    running_tsd = 0

    for smri_d, fmri_d, _ in data_loader:
        smri_d = smri_d.to(device)
        fmri_d = fmri_d.to(device)

        smri_f = smri_enc(smri_d)
        fmri_f = fmri_enc(fmri_d)
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

    train_dataset = RestJointDataset(split="train")
    val_dataset = RestJointDataset(split="dev")

    train_dataloader = DataLoader(
        val_dataset,
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

    smri_enc = SFCNEncoder(channel_number=cfg.sfcn_encoder.channel_number).to(device)
    fmri_enc = GINEncoder(
        in_channels=train_dataset.fmri.num_nodes,
        hidden_channels=cfg.gin_encoder.hidden_channels,
        num_layers=cfg.gin_encoder.num_layers,
        out_channels=cfg.gin_encoder.hidden_channels,
        dropout=cfg.gin_encoder.dropout,
        norm=cfg.gin_encoder.norm,
        emb_style=cfg.gin_encoder.emb_style,
        num_nodes=train_dataset.fmri.num_nodes,
        emb_size=cfg.gin_encoder.emb_size,
    ).to(device)

    optimizer_f = torch.optim.Adam(fmri_enc.parameters(), lr=cfg.train.lr)
    optimizer_s = torch.optim.Adam(smri_enc.parameters(), lr=cfg.train.lr)
    # criterion = MCALoss(emb_size=cfg.gin_encoder.hidden_channels)
    criterion = fmcat_loss

    # if cfg.train.lr_scheduler:
    #     scheduler1 = torch.optim.lr_scheduler.StepLR(
    #         optimizer_f, step_size=cfg.train.lr_scheduler_step_size, gamma=0.5
    #     )
    #     scheduler2 = torch.optim.lr_scheduler.StepLR(
    #         optimizer_2, step_size=cfg.train.lr_scheduler_step_size, gamma=0.5
    #     )

    for epoch in range(1, cfg.train.epochs + 1):
        start = time.time()
        train_loss, train_tsd, norm1, norm2, fgrad_norm = train_epoch(
            train_dataloader,
            criterion,
            fmri_enc,
            smri_enc,
            optimizer_f,
            optimizer_s,
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
                    "Validation loss": val_loss,
                    "Epoch time": epoch_time,
                    "Average rfnorm": norm1,
                    "Average ffnorm": norm2,
                    "Average fgradient norm": fgrad_norm,
                }
            )
        print(
            f"Epoch: {epoch}\nTrain loss: {train_loss:.4f} tsd: {train_tsd:.4f}, Val loss: {val_loss:.4f} tsd: {val_tsd:.4f}"
        )
        print(
            f"Avg. rfnorm {norm1:.4f}, Avg. ffnorm: {norm2:.4f}, Avg. fgradnorm: {fgrad_norm:.4f}"
        )


if __name__ == "__main__":
    main()
