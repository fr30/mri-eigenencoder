import hydra
import numpy as np
import torch
import wandb
import warnings

from sklearn.model_selection import KFold, train_test_split
from src.dataset import RESTfMRIDataset, GPSConcatDataset
from src.utils import EarlyStopping
from src.train import setup_experiment_gps, run_training_reg

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="configs", config_name="gps_age_abide")
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

    rest_dataset = RESTfMRIDataset(split="full", label="age")
    labels = rest_dataset.labels

    dataset = GPSConcatDataset([rest_dataset])
    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=cfg.meta.seed,
    )

    splits = kf.split(dataset, labels)
    run_train_maes = []
    run_val_maes = []
    run_train_mses = []
    run_val_mses = []
    fold_mse = []
    fold_mae = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        # Find out how many epochs should be used for training
        inner_train_idx, inner_val_idx = train_test_split(
            train_idx,
            test_size=0.1,
            random_state=42,
        )
        early_stopper = EarlyStopping(direction="min")
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

        run_training_reg(
            num_epochs=cfg.train.epochs,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=early_stopper,
            verbose=True,
            device=device,
        )
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
        train_maes, train_mses, val_maes, val_mses = run_training_reg(
            num_epochs=max_epoch + early_stopper.patience,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            verbose=True,
            device=device,
        )
        run_train_maes.append(train_maes)
        run_val_maes.append(val_maes)
        run_train_mses.append(train_mses)
        run_val_mses.append(val_mses)
        fold_mse.append(val_mses[max_epoch])
        fold_mae.append(val_maes[max_epoch])

    min_len = min(len(x) for x in run_train_mses)
    run_train_mses = [x[:min_len] for x in run_train_mses]
    run_val_mses = [x[:min_len] for x in run_val_mses]
    run_train_maes = [x[:min_len] for x in run_train_maes]
    run_val_maes = [x[:min_len] for x in run_val_maes]

    train_mse_mean = np.mean(run_train_mses, axis=0)
    val_mse_mean = np.mean(run_val_mses, axis=0)
    train_mae_mean = np.mean(run_train_maes, axis=0)
    train_mae_std = np.std(run_train_maes, axis=0)
    val_mae_mean = np.mean(run_val_maes, axis=0)
    val_mae_std = np.std(run_val_maes, axis=0)

    if cfg.wandb.enabled:
        for i in range(len(train_mse_mean)):
            run.log(
                {
                    "train_mse_mean": train_mse_mean[i],
                    "train_mae_mean": train_mae_mean[i],
                    "train_mae_std": train_mae_std[i],
                    "val_mse_mean": val_mse_mean[i],
                    "val_mae_mean": val_mae_mean[i],
                    "val_mae_std": val_mae_std[i],
                }
            )
        run.log(
            {
                "fold_mae": fold_mae,
                "fold_mse": fold_mse,
                "fold_mae_mean": np.mean(fold_mae),
                "fold_mae_std": np.std(fold_mae),
                "fold_mse_mean": np.mean(fold_mse),
                "fold_mse_std": np.std(fold_mse),
            }
        )

    print(f"Fold maes: {fold_mae}")
    print(f"Mean fold mae: {np.mean(fold_mae)}")
    print(f"Std fold mae: {np.std(fold_mae)}")
    print()
    print(f"Fold mses: {fold_mse}")
    print(f"Mean fold mse: {np.mean(fold_mse)}")
    print(f"Std fold mse: {np.std(fold_mse)}")


if __name__ == "__main__":
    main()
