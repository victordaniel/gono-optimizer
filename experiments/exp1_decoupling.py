"""
Experiment 1: Direction-Loss Decoupling Phenomenon

Demonstrates that an optimizer can exhibit near-perfect directional
consistency (cc_t -> 1) while the loss is still decreasing slowly.

Setup: 2-hidden-layer MLP on regression (y = 3x + 7 + noise),
       trained with Adam for 300 epochs.
Output: results/exp1_decoupling.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('./results', exist_ok=True)
np.random.seed(42)

# ── Data ──────────────────────────────────────────────────────────────────────
X = np.linspace(0, 10, 200).reshape(-1, 1)
y = 3 * X.squeeze() + 7 + np.random.normal(0, 4, 200)

def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)

X_n = normalize(X.squeeze())
y_n = normalize(y)

# ── MLP (NumPy) ───────────────────────────────────────────────────────────────
def relu(x):       return np.maximum(0, x)
def relu_d(x):     return (x > 0).astype(float)

class MLP:
    def __init__(self, sizes, seed=42):
        rng = np.random.RandomState(seed)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            s = np.sqrt(2.0 / sizes[i])
            self.W.append(rng.randn(sizes[i], sizes[i+1]) * s)
            self.b.append(np.zeros(sizes[i+1]))

    def forward(self, x):
        self.acts = [x]
        for W, b in zip(self.W[:-1], self.b[:-1]):
            x = relu(x @ W + b)
            self.acts.append(x)
        x = x @ self.W[-1] + self.b[-1]
        self.acts.append(x)
        return x.squeeze()

    def backward(self, x, y_true):
        n   = len(y_true)
        out = self.forward(x)
        loss = np.mean((out - y_true) ** 2)
        dout = 2 * (out - y_true) / n

        grads_W, grads_b = [], []
        delta = dout.reshape(-1, 1)
        for i in range(len(self.W) - 1, -1, -1):
            a = self.acts[i]
            grads_W.insert(0, a.T @ delta)
            grads_b.insert(0, delta.sum(axis=0))
            if i > 0:
                delta = (delta @ self.W[i].T) * relu_d(self.acts[i])
        return loss, grads_W, grads_b

    def get_flat_grad(self, x, y_true):
        _, gW, gb = self.backward(x, y_true)
        return np.concatenate([g.ravel() for g in gW + gb])

# ── Adam ──────────────────────────────────────────────────────────────────────
class Adam:
    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr=lr; self.b1=b1; self.b2=b2; self.eps=eps
        self.m={}; self.v={}; self.t=0

    def step(self, params, grads):
        self.t += 1
        updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.m:
                self.m[i] = np.zeros_like(p)
                self.v[i] = np.zeros_like(p)
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*g
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*g**2
            mh = self.m[i] / (1-self.b1**self.t)
            vh = self.v[i] / (1-self.b2**self.t)
            updated.append(p - self.lr * mh / (np.sqrt(vh) + self.eps))
        return updated

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS = 300
model  = MLP([1, 16, 8, 1])
opt    = Adam(lr=1e-3)
X_in   = X_n.reshape(-1, 1)

losses, cc_log, norm_log = [], [], []
prev_g = None

for ep in range(EPOCHS):
    loss, gW, gb = model.backward(X_in, y_n)
    flat_g = np.concatenate([g.ravel() for g in gW + gb])

    # Consecutive cosine similarity
    if prev_g is not None:
        gn  = np.linalg.norm(flat_g)
        pgn = np.linalg.norm(prev_g)
        cc  = float(np.dot(flat_g, prev_g) / (gn * pgn + 1e-10)) \
              if gn > 1e-10 and pgn > 1e-10 else 0.0
    else:
        cc = 0.0

    losses.append(loss)
    cc_log.append(cc)
    norm_log.append(np.linalg.norm(flat_g))
    prev_g = flat_g.copy()

    params = model.W + model.b
    grads  = gW + gb
    updated = opt.step(params, grads)
    n = len(model.W)
    model.W = updated[:n]
    model.b = updated[n:]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(losses, color='#E74C3C', lw=2)
axes[0].set_ylabel('Training Loss'); axes[0].grid(True, alpha=0.3)
axes[0].set_title('Direction-Loss Decoupling Phenomenon')

axes[1].plot(cc_log, color='#3498DB', lw=2)
axes[1].axhline(0.95, color='k', ls='--', lw=1, label='cc_t = 0.95')
axes[1].set_ylabel('Consecutive Cosine (cc_t)'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].semilogy(norm_log, color='#27AE60', lw=2)
axes[2].set_ylabel('Gradient Norm'); axes[2].set_xlabel('Epoch'); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./results/exp1_decoupling.png', dpi=150, bbox_inches='tight')
print("Saved: ./results/exp1_decoupling.png")

# ── Summary ───────────────────────────────────────────────────────────────────
cc_thresh = 0.95
first_stable = next((i for i in range(10, EPOCHS) if all(c > cc_thresh for c in cc_log[i:i+10])), EPOCHS)
print(f"cc_t stabilizes above {cc_thresh} at epoch {first_stable}")
print(f"Loss at epoch {first_stable}: {losses[first_stable]:.4f}")
print(f"Loss at epoch {EPOCHS-1}:     {losses[-1]:.4f}")
print(f"Loss reduction AFTER cc_t stable: {(losses[first_stable]-losses[-1])/losses[first_stable]*100:.1f}%")
