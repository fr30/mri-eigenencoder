import hydra
import time
import torch
import torch.nn.functional as F
import wandb

from src.model import GINEncoder, SFCNEncoder
from src.dataset import RestJointDataset, DataLoader
from src.eigenencoder import mca_loss
from tqdm import tqdm


def train_epoch(data_loader, fmri_enc, smri_enc, optimizer_f, optimizer_s, device):
    fmri_enc.train()
    smri_enc.train()
    running_loss = 0
    rfnorm_cum = 0
    ffnorm_cum = 0
    gradnorm_cum = 0

    for step, (smri_d, fmri_d, _) in enumerate(tqdm(data_loader)):
        smri_d = smri_d.to(device)
        fmri_d = fmri_d.to(device)

        optimizer_s.zero_grad()
        optimizer_f.zero_grad()

        smri_f = smri_enc(smri_d)
        fmri_f = fmri_enc(fmri_d)
        loss = mca_loss(fmri_f, smri_f, step + 1)

        rfnorm_cum += torch.mean(torch.norm(fmri_f, 1)).item()
        ffnorm_cum += torch.mean(torch.norm(smri_f, 1)).item()

        for param in fmri_enc.parameters():
            if param is not None:
                gradnorm_cum += torch.norm(param.grad, 2).item()

        loss.backward()
        optimizer_s.step()
        optimizer_f.step()

        running_loss += loss.item()

    cum_loss = running_loss * data_loader.batch_size / len(data_loader.dataset)
    rfnorm_avg = rfnorm_cum / len(data_loader)
    ffnorm_avg = ffnorm_cum / len(data_loader)
    gradnorm_avg = gradnorm_cum / len(data_loader)

    return cum_loss, rfnorm_avg, ffnorm_avg, gradnorm_avg


@torch.no_grad()
def test_epoch(data_loader, fmri_enc, smri_enc, device):
    fmri_enc.eval()
    smri_enc.eval()
    running_loss = 0

    for step, (smri_d, fmri_d, _) in enumerate(data_loader):
        smri_d = smri_d.to(device)
        fmri_d = fmri_d.to(device)

        smri_f = smri_enc(smri_d)
        fmri_f = fmri_enc(fmri_d)
        loss = mca_loss(fmri_f, smri_f, step + 1)

        running_loss += loss.item()

    cum_loss = running_loss * data_loader.batch_size / len(data_loader.dataset)

    return cum_loss


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
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
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

    for epoch in range(1, cfg.train.epochs + 1):
        start = time.time()
        train_loss, norm1, norm2, grad_norm = train_epoch(
            train_dataloader, fmri_enc, smri_enc, optimizer_f, optimizer_s, device
        )
        epoch_time = time.time() - start
        val_loss = test_epoch(val_dataloader, fmri_enc, smri_enc, device)
        if cfg.wandb.enabled:
            run.log(
                {
                    "Epoch": epoch,
                    "Train loss": train_loss,
                    "Validation loss": val_loss,
                    "Epoch time": epoch_time,
                    "Average rfnorm": norm1,
                    "Average ffnorm": norm2,
                    "Average gradient norm": grad_norm,
                }
            )
        print(f"Epoch: {epoch}\nTrain loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        print(f"Rfnorm avg: {norm1:.4f}, Ffnorm Avg: {norm2:.4f}")


if __name__ == "__main__":
    main()
