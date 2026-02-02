# calculating cross entropy loss
import torch

def cross_entropy_loss(logits: torch.Tensor,
                       targets: torch.Tensor) -> torch.Tensor:
    """
    logits:  Tensor of shape (N, C) containing raw scores for C classes
    targets: LongTensor of shape (N,) with values in [0, C)
    returns: single‐element Tensor with the average cross‐entropy loss
    """
    # 1) Shift logits for numerical stability: subtract max per row
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values # because it returns value, indices I think

    # 2) Compute log‐softmax: log_probs shape (N, C)
    logl1 = torch.log(torch.exp(logits) / torch.sum(torch.exp(logits), dim=-1, keepdim=True))
    # more efficient equivalent:
    # log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True) -> keep in mind: logsumexp
    # or
    # log_probs = torch.log_softmax(logits, dim=-1)
    
    # 3) Gather the log‐probs corresponding to the true classes
    # given the targets - get log-probs for the true classes?
    N = logits.size(0)
    ce_loss = logl1[range(N), targets] # why range of N? This indexes row i at column targets[i], giving you a 1D tensor of the log‐probabilities for the correct class of each sample.


    # 4) Compute the negative log‐likelihoods
    ce_loss = -ce_loss

    # 5) Average over the batch
    loss = torch.mean(ce_loss, dim=0)
    return loss
