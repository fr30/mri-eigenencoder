import torch


# def fmcat_loss(fmri_f, smri_f):
#     eps = torch.eye(fmri_f.shape[1], device=fmri_f.device) * 1e-5
#     RF = (fmri_f.T @ fmri_f) / fmri_f.shape[0] + eps
#     RG = (smri_f.T @ smri_f) / smri_f.shape[0] + eps
#     P = (fmri_f.T @ smri_f) / smri_f.shape[0]

#     lhs = torch.linalg.lstsq(RF, P).solution
#     rhs = torch.linalg.lstsq(RG, P.T).solution
#     tsd = -torch.trace(lhs @ rhs)

#     return tsd, torch.tensor(0)


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

        RF_EI = torch.linalg.pinv(RF_E, hermitian=True)
        RG_EI = torch.linalg.pinv(RG_E, hermitian=True)
        COST = (
            -RF_EI @ RF @ RF_EI @ P_E @ RG_EI @ P_E.T
            + RF_EI @ P @ RG_EI @ P_E.T
            - RF_EI @ P_E @ RG_EI @ RG @ RG_EI @ P_E.T
            + RF_EI @ P_E @ RG_EI @ P.T
        )

        RF_EI2 = torch.inverse(RF_E)
        RG_EI2 = torch.inverse(RG_E)
        TSD = RF_EI2 @ P_E @ RG_EI2 @ P_E.T

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


loss_obj = MCALoss(emb_size=128 * 4, device="cuda")


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
    P = (fmri_f.T @ smri_f) / smri_f.shape[0] + eps

    lhs = torch.linalg.lstsq(RF, P).solution
    rhs = torch.linalg.lstsq(RG, P.T).solution
    tsd = -torch.trace(lhs @ rhs)

    return tsd, tsd2.detach()
