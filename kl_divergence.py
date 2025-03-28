import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
    # distribution: mean = mu_p, std = sigma_p**2
    # N ~ (mean, std)
    # (D(P||Q)) is: [log(σ2/σ1) + (σ1^2 + (μ1-μ2)^2)/2σ2^2 - 1/2]

    norm_mean_diff = (mu_p - mu_q)**2/(sigma_q**2)
    variance_shift = (sigma_p**2)/(sigma_q**2)
    # spread / uncertainty of the distributions
    log_term = np.log(sigma_q/sigma_p)
    const = 1/2

    kl_div = log_term + const*(norm_mean_diff + variance_shift) - const

    return kl_div
