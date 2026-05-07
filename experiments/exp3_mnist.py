"""
Experiment 3: MNIST Classification Benchmark

GONO vs Adam vs AdamW vs SGD-Momentum on MNIST.
Architecture: MLP 784 -> 256 -> 128 -> 10
Output: results/exp3_mnist.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gono import GONO

os.makedirs('./results', exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── Data ──────────────────────────────────────────────────────────────────────
tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
trl = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True,  download=True, transform=tf),
    batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
tel = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=False, download=True, transform=tf),
    batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

# ── Model ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10))
    def forward(self, x): return self.net(x)

# ── Train ─────────────────────────────────────────────────────────────────────
def train(opt_name, epochs=25, seed=0):
    torch.manual_seed(seed)
    model = MLP().to(DEVICE)
    if opt_name == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif opt_name == 'AdamW':
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    elif opt_name == 'SGD-M':
        opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    elif opt_name == 'GONO':
        opt = GONO(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    accs  = []
    for ep in range(epochs):
        model.train()
        for x, y in trl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = crit(model(x), y)
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        c = tot = 0
        with torch.no_grad():
            for x, y in tel:
                x, y = x.to(DEVICE), y.to(DEVICE)
                c += model(x).argmax(1).eq(y).sum().item(); tot += y.size(0)
        acc = 100. * c / tot
        accs.append(acc)
        print(f"  [{opt_name}] ep{ep+1:3d}  acc={acc:.2f}%", flush=True)
    return accs

# ── Run ───────────────────────────────────────────────────────────────────────
OPTS   = ['SGD-M', 'Adam', 'AdamW', 'GONO']
SEEDS  = [0, 1, 2]
EPOCHS = 25

results = {}
for opt_name in OPTS:
    seed_accs = []
    for seed in SEEDS:
        print(f"\n{opt_name} seed={seed}")
        accs = train(opt_name, EPOCHS, seed)
        seed_accs.append(accs)
    results[opt_name] = seed_accs

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("MNIST RESULTS (mean ± std over 3 seeds, final epoch)")
print("="*50)
for name in OPTS:
    finals = [s[-1] for s in results[name]]
    print(f"  {name:<10}  {np.mean(finals):.2f}% ± {np.std(finals):.2f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
colors = {'SGD-M':'#95A5A6','Adam':'#E74C3C','AdamW':'#F39C12','GONO':'#3498DB'}
fig, ax = plt.subplots(figsize=(9, 5))
for name in OPTS:
    mean = np.mean(results[name], axis=0)
    std  = np.std(results[name],  axis=0)
    ep   = range(1, EPOCHS+1)
    ax.plot(ep, mean, color=colors[name], lw=2, label=f"{name} ({mean[-1]:.2f}%)")
    ax.fill_between(ep, mean-std, mean+std, color=colors[name], alpha=0.15)
ax.set_xlabel('Epoch'); ax.set_ylabel('Test Accuracy (%)')
ax.set_title('MNIST: GONO vs Baselines (mean±std, 3 seeds)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./results/exp3_mnist.png', dpi=150, bbox_inches='tight')
print("\nSaved: ./results/exp3_mnist.png")
