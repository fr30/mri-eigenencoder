import hydra
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn.functional as F
import wandb

from tqdm import tqdm
from src.model import SFCNClassifier, SFCNEncoder, DinoEncoder
from src.dataset import RESTsMRIDataset
from src.utils import CosDelayWithWarmupScheduler, IdentityScheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold


def train_epoch(data_loader, model, optimizer, scheduler, device):
    model.train()
    running_loss = 0
    correct = 0

    for x, y in tqdm(data_loader):
        x, y = x.to(device), y.to(device)
        scheduler.adjust_lr(optimizer)
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


def regular_experiment(
    cfg,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    model,
    optimizer,
    device,
):
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
        print(f"Epoch time: {epoch_time:.2f}s")

    if test_dataloader is not None:
        _, test_acc = test_epoch(test_dataloader, model, device)
        return train_accs, train_losses, val_accs, val_losses, test_acc

    return train_accs, train_losses, val_accs, val_losses, None


def kfold_experiment(cfg, run, device):
    dataset = RESTsMRIDataset(split="full")
    kf = StratifiedKFold(
        n_splits=cfg.meta.kfold_n_splits,
        shuffle=True,
        random_state=42,
    )

    splits = kf.split(dataset, dataset.sites)
    run_train_accs = []
    run_val_accs = []
    run_train_losses = []
    run_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=cfg.train.batch_size,
            # shuffle=True,
            num_workers=cfg.meta.num_workers,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        )
        val_dataloader = DataLoader(
            dataset=dataset,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.meta.num_workers,
            sampler=torch.utils.data.SubsetRandomSampler(val_idx),
        )

        if cfg.encoder.dino:
            encoder = DinoEncoder(size=cfg.encoder.dino_size).to(device)
        else:
            encoder = SFCNEncoder(
                channel_number=cfg.encoder.channel_number,
                emb_dim=cfg.encoder.emb_dim,
                norm_out=cfg.encoder.norm_out,
            ).to(device)

        if cfg.encoder.checkpoint_path is not None and not cfg.encoder.dino:
            encoder.load_state_dict(
                torch.load(cfg.encoder.checkpoint_path, weights_only=True)
            )

        if cfg.encoder.freeze:
            for param in model.encoder.parameters():
                param.requires_grad = False

        model = SFCNClassifier(
            output_dim=1,
            emb_dim=encoder.emb_dim,
            encoder=encoder,
            hidden_dim=cfg.classifier.hidden_dim,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

        print(f"Fold: {fold + 1}")
        train_accs, train_losses, val_accs, val_losses, _ = regular_experiment(
            cfg=cfg,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=None,
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


@hydra.main(version_base=None, config_path="configs", config_name="sfcn_mdd")
def main(cfg):
    if cfg.wandb.enabled:
        run = wandb.init(
            # entity=cfg.wandb.entity,
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

    if cfg.meta.kfold:
        kfold_experiment(cfg, run, device)
    else:
        train_dataset = RESTsMRIDataset(split="train")
        val_dataset = RESTsMRIDataset(split="dev")

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

        if cfg.meta.test:
            test_dataset = RESTsMRIDataset(split="test")
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=cfg.train.batch_size,
                shuffle=False,
                num_workers=cfg.meta.num_workers,
            )
        else:
            test_dataloader = None

        if cfg.encoder.dino:
            encoder = DinoEncoder(size=cfg.encoder.dino_size).to(device)
        else:
            encoder = SFCNEncoder(
                channel_number=cfg.encoder.channel_number,
                emb_dim=cfg.encoder.emb_dim,
                norm_out=cfg.encoder.norm_out,
            ).to(device)

        if cfg.encoder.checkpoint_path is not None and not cfg.encoder.dino:
            encoder.load_state_dict(
                torch.load(cfg.encoder.checkpoint_path, weights_only=True)
            )

        if cfg.encoder.checkpoint_path is not None:
            encoder.load_state_dict(
                torch.load(cfg.encoder.checkpoint_path, weights_only=True)
            )

        if cfg.encoder.freeze:
            for param in model.encoder.parameters():
                param.requires_grad = False

        model = SFCNClassifier(
            output_dim=1,
            emb_dim=encoder.emb_dim,
            encoder=encoder,
            hidden_dim=cfg.classifier.hidden_dim,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

        train_accs, train_losses, val_accs, val_losses, test_acc = regular_experiment(
            cfg=cfg,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            model=model,
            optimizer=optimizer,
            device=device,
        )

        if cfg.wandb.enabled:
            run.log(
                {
                    "train_loss": train_losses,
                    "train_acc": train_accs,
                    "val_loss": val_losses,
                    "val_acc": val_accs,
                }
            )

        if test_acc is not None:
            print(f"Test accuracy: {test_acc:.4f}")
            run.log({"test_acc": test_acc})


if __name__ == "__main__":
    main()
