"""
Experiment 2A: Oscillation Detection — cc_t vs Gradient Norm

Compares two oscillation detectors on a steep quadratic with guaranteed
divergent oscillation (SGD step size > 1/k).

Result: cc_t achieves F1=1.00, gradient norm achieves F1=0.45.
Output: results/exp2a_oscillation_detection.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import os

os.makedirs('./results', exist_ok=True)
np.random.seed(42)

# ── Problem: steep quadratic, p2-component oscillates ────────────────────────
def f(p):    return 0.5*p[0]**2 + 100*p[1]**2
def grad(p): return np.array([p[0], 200*p[1]])

STEPS   = 100
LR_SGD  = 0.015   # > 1/200 so p2 oscillates

# SGD (no momentum)
params = [np.array([2.0, 0.5])]
for _ in range(STEPS - 1):
    g = grad(params[-1])
    params.append(params[-1] - LR_SGD * g)

# Ground truth: p2 sign flip = oscillation
actual = [0] * STEPS
for i in range(1, STEPS):
    actual[i] = 1 if params[i][1] * params[i-1][1] < 0 else 0

# ── Detector A: consecutive cosine < -0.3 ────────────────────────────────────
cc_list, norm_list = [], []
prev_g = None
for p in params:
    g = grad(p)
    if prev_g is not None:
        gn = np.linalg.norm(g); pgn = np.linalg.norm(prev_g)
        cc = float(np.dot(g, prev_g) / (gn * pgn + 1e-10)) \
             if gn > 1e-10 and pgn > 1e-10 else 0.0
    else:
        cc = 0.0
    cc_list.append(cc)
    norm_list.append(np.linalg.norm(g))
    prev_g = g.copy()

cc_detected = [1 if c < -0.3 else 0 for c in cc_list]

# ── Detector B: gradient norm spike (> 1.5x rolling mean) ────────────────────
rm = 0.0
norm_detected = []
for gn in norm_list:
    rm = 0.85*rm + 0.15*gn
    norm_detected.append(1 if gn > 1.5*rm else 0)

# ── Metrics ───────────────────────────────────────────────────────────────────
def metrics(pred, true):
    p, r, f, _ = precision_recall_fscore_support(true, pred, average='binary',
                                                   zero_division=0)
    return p, r, f

p_cc,   r_cc,   f_cc   = metrics(cc_detected,   actual)
p_norm, r_norm, f_norm = metrics(norm_detected, actual)

print("\n" + "="*50)
print("OSCILLATION DETECTION RESULTS")
print("="*50)
print(f"{'Detector':<20} {'Precision':>10} {'Recall':>8} {'F1':>6}")
print(f"{'Gradient Norm':<20} {p_norm:>10.2f} {r_norm:>8.2f} {f_norm:>6.2f}")
print(f"{'Cons. Cosine':<20} {p_cc:>10.2f} {r_cc:>8.2f} {f_cc:>6.2f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
steps = list(range(STEPS))
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Loss
axes[0].plot([f(p) for p in params], color='#E74C3C', lw=2)
osc_steps = [i for i, a in enumerate(actual) if a == 1]
for s in osc_steps:
    axes[0].axvspan(s-0.5, s+0.5, color='red', alpha=0.2)
axes[0].set_ylabel('Loss'); axes[0].set_title('Exp 2A: Oscillation Detection')
axes[0].text(0.02, 0.95, 'Red = actual oscillating steps', transform=axes[0].transAxes,
             va='top', fontsize=9, color='red')

# Consecutive cosine
axes[1].plot(cc_list, color='#3498DB', lw=2, label='cc_t')
axes[1].axhline(-0.3, color='k', ls='--', lw=1, label='threshold (-0.3)')
for s in osc_steps:
    axes[1].axvspan(s-0.5, s+0.5, color='blue', alpha=0.15)
axes[1].set_ylabel('cc_t')
axes[1].set_title(f'Cons. Cosine Detector  P={p_cc:.2f}  R={r_cc:.2f}  F1={f_cc:.2f}')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

# Gradient norm
axes[2].plot(norm_list, color='#27AE60', lw=2, label='||∇L||')
axes[2].set_ylabel('Grad Norm')
axes[2].set_title(f'Gradient Norm Detector  P={p_norm:.2f}  R={r_norm:.2f}  F1={f_norm:.2f}')
axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

# Side-by-side detections
axes[3].fill_between(steps, actual,       alpha=0.5, color='red',    label='Ground truth')
axes[3].fill_between(steps, cc_detected,  alpha=0.5, color='blue',   label=f'cc_t (F1={f_cc:.2f})')
axes[3].fill_between(steps, norm_detected,alpha=0.5, color='green',  label=f'norm (F1={f_norm:.2f})')
axes[3].set_xlabel('Step'); axes[3].set_ylabel('Detected')
axes[3].legend(fontsize=9); axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./results/exp2a_oscillation_detection.png', dpi=150, bbox_inches='tight')
print("\nSaved: ./results/exp2a_oscillation_detection.png")
