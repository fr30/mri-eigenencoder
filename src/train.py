import time
import torch
import torch.nn.functional as F

from src.optim import LARS
from src.model import GPSEncoder, Classifier
from src.utils import CosDelayWithWarmupScheduler, IdentityScheduler
from torch_geometric.loader import DataLoader


def setup_experiment_gps(cfg, train_set, val_set, device):
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
        in_channels=cfg.encoder.in_channels,
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
    elif cfg.classifier.linear:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=1e-2,
            momentum=0.9,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )

    if cfg.train.lr_scheduler:
        scheduler = CosDelayWithWarmupScheduler(
            cfg.train.lr, len(train_dataloader.dataset), cfg.train.epochs
        )
    else:
        scheduler = IdentityScheduler()

    return (
        train_dataloader,
        val_dataloader,
        model,
        optimizer,
        scheduler,
    )


def run_training_mc(
    num_epochs,
    train_dataloader,
    val_dataloader,
    model,
    optimizer,
    scheduler,
    device,
    early_stopper=None,
    verbose=False,
):
    max_acc = 0
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train_loss, train_acc = train_epoch_mc(
            train_dataloader, model, optimizer, scheduler, device
        )
        epoch_time = time.time() - start
        val_loss, val_acc = test_epoch_mc(val_dataloader, model, device)

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        max_acc = max(max_acc, val_acc)

        if verbose:
            print(
                f"Epoch: {epoch}\nTrain loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}, Max acc: {max_acc:.4f}"
            )
            print(f"Epoch time: {epoch_time:.2f}s", flush=True)

        if early_stopper is not None and early_stopper.check(1 - val_acc, epoch):
            break

    return train_accs, train_losses, val_accs, val_losses


def train_epoch_mc(data_loader, model, optimizer, scheduler, device):
    model.train()
    running_loss = 0
    correct = 0

    for data in data_loader:
        data = data.to(device)
        scheduler.adjust_lr(optimizer)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        correct += (out.argmax(dim=1) == data.y).sum().item()
        running_loss += loss.item()

    cum_loss = running_loss * data_loader.batch_size / len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    return cum_loss, acc


@torch.no_grad()
def test_epoch_mc(data_loader, model, device):
    model.eval()
    running_loss = 0
    correct = 0

    for data in data_loader:
        data = data.to(device)
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        running_loss += loss.item()
        correct += (out.argmax(dim=1) == data.y).sum().item()

    cum_loss = running_loss * data_loader.batch_size / len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    return cum_loss, acc


def run_training_bc(
    num_epochs,
    train_dataloader,
    val_dataloader,
    model,
    optimizer,
    scheduler,
    device,
    early_stopper=None,
    verbose=False,
):
    max_acc = 0
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train_loss, train_acc = train_epoch_bc(
            train_dataloader, model, optimizer, scheduler, device
        )
        epoch_time = time.time() - start
        val_loss, val_acc = test_epoch_bc(val_dataloader, model, device)

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        max_acc = max(max_acc, val_acc)

        if verbose:
            print(
                f"Epoch: {epoch}\nTrain loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}, Max acc: {max_acc:.4f}"
            )
            print(f"Epoch time: {epoch_time:.2f}s", flush=True)

        if early_stopper is not None and early_stopper.check(1 - val_acc, epoch):
            break

    return train_accs, train_losses, val_accs, val_losses


def train_epoch_bc(data_loader, model, optimizer, scheduler, device):
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
def test_epoch_bc(data_loader, model, device):
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
