import torch
import math
from typing import Optional, Callable, Iterable

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError("Learning rate must be positive")
        super().__init__(
            params, 
            defaults={
                "lr": lr, 
                "betas": betas,
                "eps": eps, 
                "weight_decay": weight_decay,
            })
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                g = p.grad
                
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                if "v" not in state:
                    state["v"] = torch.zeros_like(p)
                if "t" not in state:
                    state["t"] = 1
                
                m = state["m"]
                v = state["v"]
                t = state["t"]

                # m = beta1 * m + (1 - beta1) * g
                # v = beta2 * v + (1 - beta2) * g**2
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # p.data -= lr_t * m / (v.sqrt() + eps)
                # p.data -= lr * weight_decay * p.data
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)
                p.data.add_(p.data, alpha=-lr * weight_decay)
                
                state["t"] = t + 1
        return loss

def learning_rate_schedule(t, max_lr, min_lr, T_w, T_c):
    if t < T_w:
        return max_lr * t / T_w
    elif T_w <= t and t <= T_c:
        return min_lr + (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) * (max_lr - min_lr) / 2
    else:
        return min_lr

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    l2_norm = 0.0
    for p in parameters:
        g = p.grad
        if g is not None:
            l2_norm += (g**2).sum()
    l2_norm = math.sqrt(l2_norm)
    if l2_norm > max_l2_norm:
        for p in parameters:
            g = p.grad
            if g is not None:
                p.grad.data *= max_l2_norm / (l2_norm + 1e-6)