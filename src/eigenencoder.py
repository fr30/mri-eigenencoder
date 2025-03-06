import torch


def fmcat_loss(fmri_f, smri_f):
    eps = torch.eye(fmri_f.shape[1], device=fmri_f.device) * 1e-6
    RF = (fmri_f.T @ fmri_f) / fmri_f.shape[0] + eps
    RG = (smri_f.T @ smri_f) / smri_f.shape[0] + eps
    P = (fmri_f.T @ smri_f) / smri_f.shape[0] + eps

    lhs = torch.linalg.solve(RF, P)
    rhs = torch.linalg.solve(RG, P.T)
    tsd_m = lhs @ rhs
    tsd = -torch.trace(tsd_m / torch.norm(tsd_m, p=2).detach())

    # eye = torch.eye(lhs.shape[0]).to(lhs.device)
    # MF = fmri_f.T @ fmri_f
    # MG = smri_f.T @ smri_f
    # reg = torch.square(MF.T @ MF - eye).sum() + torch.square(MG.T @ MG - eye).sum()

    return tsd, torch.tensor(1)

    return tsd, torch.tensor(0)

    # print(lhs.max().item(), rhs.max().item())
    # print(tsd.item(), torch.trace(lhs @ rhs).item())
    # tsd = -torch.trace(lhs @ rhs)
    lhs.retain_grad()

    rhs.retain_grad()

    print(rhs)
    tsd.backward()
    print(rhs.grad)
    print(rhs.grad.max())
    exit(0)

    return -torch.trace(tsd), torch.tensor(0)


class MCALoss:
    def __init__(self, emb_size):
        self.track_cov_final = torch.zeros((emb_size * 2)).cuda()
        self.track_cov_estimate_final = torch.zeros((emb_size * 2)).cuda()
        self.step = 1

    def __call__(self, fmri_f, smri_f):
        return self.mca_loss(fmri_f, smri_f)

    def mca_loss(self, fmri_f, smri_f):
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
            RFG, self.track_cov_final, self.step, emb_size
        )
        cost, tsd = self.cost_trace(RFG, self.track_cov_estimate_final, emb_size)

        return cost, tsd

    @staticmethod
    def cost_trace(RFG, track_cov_estimate_final, dim):
        RF_E = track_cov_estimate_final[:dim, :dim]
        RG_E = track_cov_estimate_final[dim:, dim:]
        P_E = track_cov_estimate_final[:dim, dim:]

        RF_EI = torch.inverse(RF_E)
        RG_EI = torch.inverse(RG_E)

        RF = RFG[:dim, :dim]
        RG = RFG[dim:, dim:]
        P = RFG[:dim, dim:]

        COST = (
            -RF_EI @ RF @ RF_EI @ P_E @ RG_EI @ P_E.T
            + RF_EI @ P @ RG_EI @ P_E.T
            - RF_EI @ P_E @ RG_EI @ RG @ RG_EI @ P_E.T
            + RF_EI @ P_E @ RG_EI @ P.T
        )
        # TSD = RF_EI @ P_E @ RG_EI @ P_E.T
        tsd = torch.linalg.solve(RF_E, P_E) @ torch.linalg.solve(RG_E, P_E.T)
        # print("==============================")
        # print(TSD)
        # print(tsd)
        # print(torch.abs(TSD - tsd).sum().item())
        # x1 = RF_EI @ P_E
        # x2 = torch.linalg.solve(RF_E, P_E)
        # print(x1)
        # print(x2)
        # print(torch.abs(x1 - x2).sum().item())
        # # x2 = RG_EI @ P_E.T
        # exit(0)
        # return -torch.trace(COST), -torch.trace(TSD)
        return -torch.trace(COST), -torch.trace(tsd)

    @staticmethod
    def calc_track_cov(RP, track_cov, step, dim):
        device = RP.device
        cov = RP + torch.eye((RP.shape[0])).to(device) * 1e-6
        track_cov, cov_estimate = MCALoss.adaptive_estimation(track_cov, 0.5, cov, step)

        # cov_estimate_f = cov_estimate[:dim, :dim]
        # cov_f = cov[:dim, :dim]

        # cov_estimate_g = cov_estimate[dim:, dim:]
        # cov_g = cov[dim:, dim:]

        # LOSS = (
        #     (torch.linalg.inv(cov_estimate) * cov).sum()
        #     - (torch.linalg.inv(cov_estimate_f) * cov_f).sum()
        #     - (torch.linalg.inv(cov_estimate_g) * cov_g).sum()
        # )
        # return track_cov, cov_estimate, LOSS
        return track_cov, cov_estimate

    @staticmethod
    def adaptive_estimation(v_t, beta, square_term, i):
        v_t = beta * v_t + (1 - beta) * square_term.detach()
        return v_t, (v_t / (1 - beta**i))
