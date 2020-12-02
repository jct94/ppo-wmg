# miscellaneous utils
import torch

def get_grad(model):
    # gets accumulated gradients in model parameters as a single vector
    pl = []
    for p in model.parameters():
        pl.append(p.grad.reshape(-1))
    return torch.cat(pl, 0)

def get_grad_norm(model):
    grad = get_grad(model)
    return grad.pow(2).sum(-1).pow(0.5)

# get cosine similarity matrix of a multi-slot model

def memo_cosine_sim_matrix(memos):
    vec_list = []
