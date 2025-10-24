import torch
import numpy.typing as npt
import random
import os
from typing import BinaryIO, IO

def top_p_sampling(probs: torch.Tensor, top_p: float=1.0) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_cumsum = torch.cumsum(probs_sort, dim=-1)
    probs_mask = probs_cumsum > top_p
    probs_mask[0] = False
    probs_sort = probs_sort.masked_fill(probs_mask, 0.0)
    probs = torch.zeros_like(probs).scatter(dim=-1, index=probs_idx, src=probs_sort)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs

def softmax(x: torch.Tensor, dim: int=-1, temperature: float=1.0) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    # return torch.exp(x / temperature) / torch.exp(x / temperature).sum(dim=dim, keepdim=True)
    exp_x = torch.exp(x / temperature)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def silu(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + torch.exp(-x))
    
def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # inputs = inputs - inputs.max(dim=-1, keepdim=True).values
    # exp_sum_log = inputs.exp().sum(dim=-1).log()
    exp_sum_log = torch.logsumexp(inputs, dim=-1)
    pred_probs = inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (exp_sum_log - pred_probs).mean()

def perplexity(losses: torch.Tensor) -> torch.Tensor:
    return losses.mean().exp()

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    idxs = torch.randint(0, len(dataset) - context_length, (batch_size,1))
    idxs = torch.cat([idxs + i for i in range(context_length)], dim=-1)
    batch = torch.from_numpy(dataset[idxs]).to(device)
    target = torch.from_numpy(dataset[idxs+1]).to(device)
    return batch, target

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    obj = {}
    obj["model_state_dict"] = model.state_dict()
    obj["optimizer_state_dict"] = optimizer.state_dict()
    obj["iteration"] = iteration
    torch.save(obj, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj["model_state_dict"])
    optimizer.load_state_dict(obj["optimizer_state_dict"])
    return obj["iteration"]