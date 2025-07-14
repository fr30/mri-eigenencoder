import hydra
import numpy as np
import torch
import wandb
import warnings

from sklearn.model_selection import StratifiedKFold, train_test_split
from src.dataset import ADHD200fMRIDataset, GPSConcatDataset
from src.utils import EarlyStopping
from src.train import setup_experiment_gps, run_training_bc

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="configs", config_name="gps_adhd")
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

    adhd_dataset = ADHD200fMRIDataset(split="full", label="adhd")
    labels = adhd_dataset.labels

    dataset = GPSConcatDataset([adhd_dataset])
    kf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=cfg.meta.seed,
    )

    splits = kf.split(dataset, labels)
    run_train_accs = []
    run_val_accs = []
    run_train_losses = []
    run_val_losses = []
    fold_acc = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        # Find out how many epochs should be used for training
        inner_train_idx, inner_val_idx = train_test_split(
            train_idx,
            test_size=0.2,
            random_state=42,
            stratify=labels[train_idx],
        )
        early_stopper = EarlyStopping()
        inner_train_set = torch.utils.data.dataset.Subset(dataset, inner_train_idx)
        inner_val_set = torch.utils.data.dataset.Subset(dataset, inner_val_idx)

        train_dataloader, val_dataloader, model, optimizer, scheduler = (
            setup_experiment_gps(
                cfg=cfg,
                train_set=inner_train_set,
                val_set=inner_val_set,
                device=device,
            )
        )

        run_training_bc(
            num_epochs=cfg.train.epochs,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=early_stopper,
            verbose=False,
            device=device,
        )
        # max_epoch = cfg.train.epochs
        max_epoch = early_stopper.best_epoch

        # Run actual training with the best number of epochs
        train_set = torch.utils.data.dataset.Subset(dataset, train_idx)
        val_set = torch.utils.data.dataset.Subset(dataset, val_idx)

        train_dataloader, val_dataloader, model, optimizer, scheduler = (
            setup_experiment_gps(
                cfg=cfg,
                train_set=train_set,
                val_set=val_set,
                device=device,
            )
        )
        print(f"Fold: {fold + 1}, Max epoch: {max_epoch}")
        train_accs, train_losses, val_accs, val_losses = run_training_bc(
            # num_epochs=max_epoch,
            num_epochs=max_epoch + early_stopper.patience,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            verbose=True,
            device=device,
        )
        run_train_accs.append(train_accs)
        run_val_accs.append(val_accs)
        run_train_losses.append(train_losses)
        run_val_losses.append(val_losses)
        fold_acc.append(val_accs[max_epoch - 1])

    min_len = min(len(x) for x in run_train_losses)
    run_train_losses = [x[:min_len] for x in run_train_losses]
    run_val_losses = [x[:min_len] for x in run_val_losses]
    run_train_accs = [x[:min_len] for x in run_train_accs]
    run_val_accs = [x[:min_len] for x in run_val_accs]

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
                    "train_loss_mean": train_loss_mean[i],
                    "train_acc_mean": train_acc_mean[i],
                    "train_acc_std": train_acc_std[i],
                    "val_loss_mean": val_loss_mean[i],
                    "val_acc_mean": val_acc_mean[i],
                    "val_acc_std": val_acc_std[i],
                }
            )
        run.log(
            {
                "fold_acc": fold_acc,
                "fold_acc_mean": np.mean(fold_acc),
                "fold_acc_std": np.std(fold_acc),
            }
        )

    print(f"Fold accuracies: {fold_acc}")
    print(f"Mean fold accuracy: {np.mean(fold_acc)}")
    print(f"Std fold accuracy: {np.std(fold_acc)}")


if __name__ == "__main__":
    main()
