import numpy as np


class CosDelayWithWarmupScheduler:
    def __init__(self, base_lr, loader_len, num_epochs):
        self.base_lr = base_lr
        self.lr_weights = 0.2
        self.lr_biases = 0.0048
        self.loader_len = loader_len
        self.num_epochs = num_epochs
        self.step = 0

    def adjust_lr(self, optimizer):
        max_steps = self.num_epochs * self.loader_len
        warmup_steps = 10 * self.loader_len

        if self.step < warmup_steps:
            lr = self.base_lr * self.step / warmup_steps
        else:
            self.step -= warmup_steps
            max_steps -= warmup_steps
            q = 0.5 * (1 + np.cos(np.pi * self.step / max_steps))
            end_lr = self.base_lr * 0.001
            lr = self.base_lr * q + end_lr * (1 - q)

        # print(len(optimizer.param_groups))
        optimizer.param_groups[0]["lr"] = lr * self.lr_weights
        # optimizer.param_groups[1]['lr'] = lr * self.lr_biases
        self.step += 1


class IdentityScheduler:
    @classmethod
    def adjust_lr(self, _):
        pass
