import torch


def mca_loss(fmri_f, smri_f, step):
    device = fmri_f.device
    emb_size = fmri_f.shape[1]
    track_cov_final = torch.zeros((emb_size + emb_size)).to(device)
    track_cov_estimate_final = torch.zeros((emb_size + emb_size)).to(device)

    RF = (fmri_f.T @ fmri_f) / fmri_f.shape[0]
    RG = (smri_f.T @ smri_f) / smri_f.shape[0]
    P = (fmri_f.T @ smri_f) / smri_f.shape[0]

    input_dim, output_dim = RF.shape[1], RG.shape[1]
    RFG = torch.zeros((input_dim + output_dim, input_dim + output_dim)).to(
        fmri_f.device
    )
    RFG[:input_dim, :input_dim] = RF
    RFG[input_dim:, input_dim:] = RG
    RFG[:input_dim, input_dim:] = P
    RFG[input_dim:, :input_dim] = P.T

    track_cov_final, track_cov_estimate_final = calc_track_cov(
        RFG, track_cov_final, step, emb_size
    )
    cost = cost_trace(RFG, track_cov_estimate_final, emb_size)

    return cost


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
    # return -torch.trace(COST), -torch.trace(TSD)
    return -torch.trace(COST)


def calc_track_cov(RP, track_cov, step, dim):
    device = RP.device
    cov = RP + torch.eye((RP.shape[0])).to(device) * 1e-6
    track_cov, cov_estimate = adaptive_estimation(track_cov, 0.5, cov, step)

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


def adaptive_estimation(v_t, beta, square_term, i):
    v_t = beta * v_t + (1 - beta) * square_term.detach()
    return v_t, (v_t / (1 - beta**i))
