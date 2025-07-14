import torch
import torch.nn.functional as F


class HFMCALoss:
    def __init__(self, device):
        self.device = device

    def __call__(self, y, y_dash):
        g_patches = y.unsqueeze(3).unsqueeze(2).unsqueeze(2)
        f_proj = y_dash.unsqueeze(2).unsqueeze(2)

        f1 = torch.flatten(f_proj.permute(0, 2, 3, 1), 0, -2)
        f2 = g_patches.permute(0, 2, 3, 4, 5, 1).flatten(0, -2)
        f3 = torch.flatten(g_patches.mean(dim=(-1, -2)).permute(0, 2, 3, 1), 0, -2)
        input_dim, output_dim = f_proj.shape[1], g_patches.shape[1]

        P = f1.T @ f3 / f1.shape[0]
        RF = f1.T @ f1 / f1.shape[0]
        RG = f2.T @ f2 / f2.shape[0]
        RFG = torch.zeros((input_dim + output_dim, input_dim + output_dim)).to(
            self.device
        )
        RFG[:input_dim, :input_dim] = RF
        RFG[input_dim:, input_dim:] = RG
        RFG[:input_dim, input_dim:] = P
        RFG[input_dim:, :input_dim] = P.T

        RFG = RFG + torch.eye((RFG.shape[0])).to(self.device) * 1e-3
        RF = RF + torch.eye((RF.shape[0])).to(self.device) * 1e-3
        RG = RG + torch.eye((RG.shape[0])).to(self.device) * 1e-3

        return torch.logdet(RFG) - torch.logdet(RF) - torch.logdet(RG)


class MCALoss:
    def __init__(self, emb_size, device):
        self.track_cov_final = torch.zeros((emb_size * 2, emb_size * 2)).to(device)
        self.track_cov_estimate_final = torch.zeros((emb_size * 2, emb_size * 2)).to(
            device
        )
        self.step = 0

    def __call__(self, fmri_f, smri_f):
        return self.mca_loss(fmri_f, smri_f)

    def mca_loss(self, fmri_f, smri_f):
        self.step += 1
        device = fmri_f.device
        emb_size = fmri_f.shape[1]

        RF = (fmri_f.T @ fmri_f) / fmri_f.shape[0]
        RG = (smri_f.T @ smri_f) / smri_f.shape[0]
        P = (fmri_f.T @ smri_f) / smri_f.shape[0]

        input_dim, output_dim = RF.shape[1], RG.shape[1]
        RFG = torch.zeros((input_dim + output_dim, input_dim + output_dim)).to(device)
        RFG[:input_dim, :input_dim] = RF
        RFG[input_dim:, input_dim:] = RG
        RFG[:input_dim, input_dim:] = P
        RFG[input_dim:, :input_dim] = P.T

        self.track_cov_final, self.track_cov_estimate_final = self.calc_track_cov(
            RFG, self.track_cov_final, self.step
        )
        cost, tsd = self.cost_trace(RFG, self.track_cov_estimate_final, emb_size)
        return cost, tsd

    @staticmethod
    def cost_trace(RFG, track_cov_estimate_final, dim):
        RF_E = track_cov_estimate_final[:dim, :dim]
        RG_E = track_cov_estimate_final[dim:, dim:]
        P_E = track_cov_estimate_final[:dim, dim:]

        RF = RFG[:dim, :dim]
        RG = RFG[dim:, dim:]
        P = RFG[:dim, dim:]

        RF_EI = torch.inverse(RF_E)
        RG_EI = torch.inverse(RG_E)
        COST = (
            -RF_EI @ RF @ RF_EI @ P_E @ RG_EI @ P_E.T
            + RF_EI @ P @ RG_EI @ P_E.T
            - RF_EI @ P_E @ RG_EI @ RG @ RG_EI @ P_E.T
            + RF_EI @ P_E @ RG_EI @ P.T
        )

        # RF_EI2 = torch.inverse(RF_E)
        # RG_EI2 = torch.inverse(RG_E)
        TSD = RF_EI @ P_E @ RG_EI @ P_E.T

        return -torch.trace(COST), -torch.trace(TSD).detach()

    @staticmethod
    def calc_track_cov(RP, track_cov, step):
        device = RP.device
        cov = RP + torch.eye((RP.shape[0])).to(device) * 1e-6
        track_cov, cov_estimate = MCALoss.adaptive_estimation(track_cov, 0.5, cov, step)
        return track_cov, cov_estimate

    @staticmethod
    def adaptive_estimation(v_t, beta, square_term, i):
        v_t = beta * v_t + (1 - beta) * square_term.detach()
        return v_t, (v_t / (1 - beta**i))


loss_obj = MCALoss(emb_size=256 * 4, device="cuda")


def fmcat_loss(fmri_f, smri_f):
    global loss_obj
    # eps = torch.eye(fmri_f.shape[1], device=fmri_f.device) * 1e-5
    # RF = (fmri_f.T @ fmri_f) / fmri_f.shape[0] + eps
    # RG = (smri_f.T @ smri_f) / smri_f.shape[0] + eps
    # P = (fmri_f.T @ smri_f) / smri_f.shape[0] + eps

    # RF_norm = torch.norm(RF).detach()
    # RG_norm = torch.norm(RG).detach()
    # lhs = torch.linalg.solve(RF / RF_norm, P / RF_norm)
    # rhs = torch.linalg.solve(RG / RG_norm, P.T / RG_norm)
    # tsd2 = -torch.trace(lhs @ rhs)
    # tsd = -torch.trace(lhs @ rhs)
    tsd2 = loss_obj(fmri_f, smri_f)[1]

    eps = torch.eye(fmri_f.shape[1], device=fmri_f.device) * 1e-5
    RF = (fmri_f.T @ fmri_f) / fmri_f.shape[0] + eps
    RG = (smri_f.T @ smri_f) / smri_f.shape[0] + eps
    P = (fmri_f.T @ smri_f) / smri_f.shape[0]

    lhs = torch.linalg.lstsq(RF, P).solution
    rhs = torch.linalg.lstsq(RG, P.T).solution
    tsd = -torch.trace(lhs @ rhs)

    return tsd, tsd2.detach()


# Implementation from https://github.com/MaxLikesMath/Barlow-Twins-Pytorch
class BTLoss:
    def __init__(self, batch_size, accelerator=None, lambd=0.0051):
        self.batch_size = batch_size
        self.accelerator = accelerator
        self.lambd = lambd

    def __call__(self, z1, z2):
        c = z1.T @ z2

        if self.accelerator is not None:
            c = self.accelerator.reduce(c)

        c /= self.batch_size
        on_diag = (1 - torch.diagonal(c)).pow(2).sum()
        off_diag = c[~torch.eye(c.shape[0], dtype=bool)].pow(2).sum()
        loss = on_diag + self.lambd * off_diag

        return loss

    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# Implementation from https://github.com/sthalles/SimCLR/
class InfoNCELoss:
    def __init__(self, device, temperature=0.07):
        self.temperature = temperature
        self.device = device
        self.n_views = 2

    def __call__(self, features):
        batch_size = features.shape[0] // self.n_views
        labels = torch.cat(
            [torch.arange(batch_size) for i in range(self.n_views)],
            dim=0,
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        logits = logits / self.temperature
        loss = F.cross_entropy(logits, labels)

        return loss


class InfoNCELoss:
    def __init__(self, device, temperature=0.07):
        self.temperature = temperature
        self.device = device
        self.n_views = 2

    def __call__(self, features):
        batch_size = features.shape[0] // self.n_views
        labels = torch.cat(
            [torch.arange(batch_size) for i in range(self.n_views)],
            dim=0,
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        logits = logits / self.temperature
        loss = F.cross_entropy(logits, labels)

        return loss


class VicregLoss:
    def __init__(
        self, num_features, batch_size, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0
    ):
        self.num_features = num_features
        self.batch_size = batch_size
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def __call__(self, x1, x2):
        repr_loss = F.mse_loss(x1, x2)

        x1 = x1 - x1.mean(dim=0)
        x2 = x2 - x2.mean(dim=0)

        std_x1 = torch.sqrt(x1.var(dim=0) + 0.0001)
        std_x2 = torch.sqrt(x2.var(dim=0) + 0.0001)
        std_loss = (
            torch.mean(F.relu(1 - std_x1)) / 2 + torch.mean(F.relu(1 - std_x2)) / 2
        )

        cov_x1 = (x1.T @ x1) / (self.batch_size - 1)
        cov_x2 = (x2.T @ x2) / (self.batch_size - 1)
        cov_loss = self._off_diagonal(cov_x1).pow_(2).sum().div(
            self.num_features
        ) + self._off_diagonal(cov_x2).pow_(2).sum().div(self.num_features)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss

    def _off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
