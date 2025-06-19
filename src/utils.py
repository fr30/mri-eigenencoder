# import elasticdeform
import numpy as np
import torch

from scipy.ndimage.interpolation import affine_transform
from torch_geometric.utils import dense_to_sparse


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

        optimizer.param_groups[0]["lr"] = lr * self.lr_weights

        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]["lr"] = lr * self.lr_biases

        self.step += 1


class IdentityScheduler:
    @classmethod
    def adjust_lr(self, _):
        pass


def patch_extraction(Xb, sizePatches=128, Npatches=1):
    """
    3D patch extraction
    """

    batch_size = len(Xb)
    channels, rows, columns, slices = Xb[0].shape
    X_patches = np.empty(
        (batch_size * Npatches, channels, sizePatches, sizePatches, sizePatches)
    )
    # y_patches = np.empty((batch_size * Npatches, sizePatches, sizePatches, sizePatches))
    i = 0
    for b in range(batch_size):
        for p in range(Npatches):
            x = np.random.randint(rows - sizePatches + 1)
            y = np.random.randint(columns - sizePatches + 1)
            z = np.random.randint(slices - sizePatches + 1)

            X_patches[i] = Xb[b][
                :, x : x + sizePatches, y : y + sizePatches, z : z + sizePatches
            ]
            # y_patches[i] = yb[
            #     b, x : x + sizePatches, y : y + sizePatches, z : z + sizePatches
            # ]
            i += 1

    return X_patches


def flip3D(X):
    """
    Flip the 3D image respect one of the 3 axis chosen randomly
    """
    choice = np.random.randint(3)
    if choice == 0:  # flip on x
        X_flip = X[::-1, :, :, :]
    if choice == 1:  # flip on y
        X_flip = X[:, ::-1, :, :]
    if choice == 2:  # flip on z
        X_flip = X[:, :, ::-1, :]

    return X_flip


def rotation_zoom3D(X):
    """
    Rotate a 3D image with alfa, beta and gamma degree respect the axis x, y and z respectively.
    The three angles are chosen randomly between 0-30 degrees
    """
    alpha, beta, gamma = np.random.random_sample(3) * np.pi / 2
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ]
    )

    Ry = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )

    Rz = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )

    R_rot = np.dot(np.dot(Rx, Ry), Rz)

    a, b = 0.8, 1.2
    alpha, beta, gamma = (b - a) * np.random.random_sample(3) + a
    R_scale = np.array([[alpha, 0, 0], [0, beta, 0], [0, 0, gamma]])

    R = np.dot(R_rot, R_scale)
    X_rot = np.empty_like(X)

    for channel in range(X.shape[-1]):
        X_rot[:, :, :, channel] = affine_transform(
            X[:, :, :, channel], R, offset=0, order=1, mode="constant"
        )

    return X_rot


def brightness(X):
    """
    Changing the brighness of a image using power-law gamma transformation.
    Gain and gamma are chosen randomly for each image channel.

    Gain chosen between [0.8 - 1.2]
    Gamma chosen between [0.8 - 1.2]

    new_im = gain * im^gamma
    """

    X_new = np.zeros(X.shape)
    for c in range(X.shape[-1]):
        im = X[:, :, :, c]
        gain, gamma = (1.2 - 0.8) * np.random.random_sample(
            2,
        ) + 0.8
        im_new = np.sign(im) * gain * (np.abs(im) ** gamma)
        X_new[:, :, :, c] = im_new

    return X_new


def elastic(X):
    """
    Elastic deformation on a image and its target
    """
    [Xel] = elasticdeform.deform_random_grid(
        [X], sigma=2, axis=[(1, 2, 3)], order=[1], mode="constant"
    )

    return Xel


def random_decisions(N):
    """
    Generate N random decisions for augmentation
    N should be equal to the batch size
    """

    decisions = np.zeros(
        (N, 4)
    )  # 4 is number of aug techniques to combine (patch extraction excluded)
    for n in range(N):
        decisions[n] = np.random.randint(2, size=4)

    return decisions


def combine_aug(X, do):
    """
    Combine randomly the different augmentation techniques written above
    """
    Xnew = X

    # make sure to use at least 25% of original images
    if np.random.random_sample() > 0.75:
        return Xnew
    else:
        if do[0] == 1:
            Xnew = flip3D(Xnew)

        if do[1] == 1:
            Xnew = brightness(Xnew)

        if do[2] == 1:
            Xnew = rotation_zoom3D(Xnew)

        if do[3] == 1:
            Xnew = elastic(Xnew)

        return Xnew


def aug_batch(Xb):
    """
    Generate a augmented image batch
    """
    batch_size = len(Xb)
    newXb = np.empty_like(Xb)

    decisions = random_decisions(batch_size)

    x_new = [combine_aug(X, do) for X, do in zip(Xb, decisions)]
    return x_new
    # pool = mp.Pool(processes=8)
    # multi_result = pool.starmap(combine_aug, inputs)
    # pool.close()

    for i in range(batch_size):
        newXb[i] = multi_result[i][0]

    return newXb


def create_corr(data):
    eps = 1e-16
    data = data - np.nanmean(data, axis=0, keepdims=True)
    data = data / (np.nanstd(data, axis=0, keepdims=True) + eps)

    R = np.corrcoef(data)
    R[np.isnan(R)] = 0
    R = R - np.diag(np.diag(R))
    R[R >= 1] = 1 - eps
    corr = 0.5 * np.log((1 + R) / (1 - R))

    return corr


def corr_to_graph(corr):
    node_features = torch.from_numpy(corr).to(torch.float32)
    topk = node_features.reshape(-1)
    topk, _ = torch.sort(abs(topk), dim=0, descending=True)
    threshold = topk[int(node_features.shape[0] ** 2 / 20 * 2)]
    adj = (torch.abs(node_features) >= threshold).to(int)
    edge_index = dense_to_sparse(adj)[0]

    # if num_edges is None:
    #     num_edges = edge_index.shape[1]

    # edge_index = edge_index[:, :num_edges]

    return node_features, edge_index
