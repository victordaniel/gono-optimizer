"""
GONO: Gradient-Oriented Norm-Adaptive Optimizer

Adapts Adam's momentum coefficient beta1 based on consecutive gradient
cosine similarity (cc_t), treating directional alignment as a first-class
optimization signal.

Reference:
    "Directional Consistency as a Complementary Optimization Signal:
     The GONO Framework", arXiv 2026.
"""

import math
import torch


class GONO(torch.optim.Optimizer):
    """GONO optimizer: adaptive beta1 via consecutive cosine similarity.

    Args:
        params:     iterable of parameters to optimize
        lr:         learning rate (default: 1e-3)
        beta1:      base momentum coefficient (default: 0.9)
        beta2:      second moment coefficient (default: 0.999)
        eps:        numerical stability term (default: 1e-8)
        lam:        sensitivity of beta1 to cc_t (default: 0.4)
        beta1_min:  minimum allowed beta1 (default: 0.5)
        beta1_max:  maximum allowed beta1 (default: 0.99)
        weight_decay: L2 regularization (default: 0.0)

    Algorithm (Algorithm 1 from paper):
        cc_t  = <g_t, g_{t-1}> / (||g_t|| * ||g_{t-1}|| + eps)
        b1_t  = clip(beta1 * (1 + lam * cc_t), beta1_min, beta1_max)
        m_t   = b1_t * m_{t-1} + (1 - b1_t) * g_t
        v_t   = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        theta = theta - lr * sqrt(1-beta2^t)/(1-beta1^t) * m_t / (sqrt(v_t) + eps)

    Reduces exactly to Adam when lam=0.
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 lam=0.4, beta1_min=0.5, beta1_max=0.99, weight_decay=0.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        lam=lam, beta1_min=beta1_min, beta1_max=beta1_max,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr       = group['lr']
            b1       = group['beta1']
            b2       = group['beta2']
            eps      = group['eps']
            lam      = group['lam']
            b1_min   = group['beta1_min']
            b1_max   = group['beta1_max']
            wd       = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state['t']      = 0
                    state['m']      = torch.zeros_like(p)
                    state['v']      = torch.zeros_like(p)
                    state['g_prev'] = torch.zeros_like(p)

                state['t'] += 1
                t = state['t']
                m, v, g_prev = state['m'], state['v'], state['g_prev']

                # Consecutive cosine similarity (one dot product, O(d))
                if t > 1:
                    gn  = g.norm().item()
                    pgn = g_prev.norm().item()
                    if gn > 1e-10 and pgn > 1e-10:
                        cc = (g * g_prev).sum().item() / (gn * pgn + 1e-10)
                    else:
                        cc = 0.0
                else:
                    cc = 0.0

                # Adaptive beta1
                b1t = float(torch.tensor(b1 * (1.0 + lam * cc)).clamp(b1_min, b1_max))

                # Moment updates
                m.mul_(b1t).add_(g, alpha=1.0 - b1t)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)

                # Bias correction (uses static beta1 as in paper)
                bias1 = 1.0 - b1 ** t
                bias2 = 1.0 - b2 ** t
                step_size = lr * math.sqrt(bias2) / bias1

                p.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)
                g_prev.copy_(g)

        return loss
