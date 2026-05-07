"""
Experiment 4: CIFAR-10 MLP Benchmark

GONO vs Adam vs AdamW vs SGD-Momentum on CIFAR-10 (10k subset).
Architecture: MLP 3072 -> 256 -> 128 -> 10
Output: results/exp4_cifar10.png
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

# ── Data (10k training subset) ────────────────────────────────────────────────
tf_tr = T.Compose([T.ToTensor(),
                   T.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
tf_te = T.Compose([T.ToTensor(),
                   T.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])

full_train = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=tf_tr)
subset     = torch.utils.data.Subset(full_train, range(10000))
trl = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=True,  num_workers=2)
tel = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=tf_te),
    batch_size=256, shuffle=False, num_workers=2)

# ── Model ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 256), nn.ReLU(),
            nn.Linear(256,  128), nn.ReLU(),
            nn.Linear(128,  10))
    def forward(self, x): return self.net(x)

# ── Train ─────────────────────────────────────────────────────────────────────
def train(opt_name, epochs=20, seed=0):
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
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for x, y in trl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = crit(model(x), y)
            loss.backward(); opt.step()
    model.eval()
    c = tot = 0
    with torch.no_grad():
        for x, y in tel:
            x, y = x.to(DEVICE), y.to(DEVICE)
            c += model(x).argmax(1).eq(y).sum().item(); tot += y.size(0)
    acc = 100. * c / tot
    print(f"  [{opt_name}] seed={seed}  acc={acc:.2f}%", flush=True)
    return acc

# ── Run ───────────────────────────────────────────────────────────────────────
OPTS  = ['SGD-M', 'Adam', 'AdamW', 'GONO']
SEEDS = [0, 1, 2]

results = {}
for opt_name in OPTS:
    accs = [train(opt_name, 20, s) for s in SEEDS]
    results[opt_name] = accs

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("CIFAR-10 MLP RESULTS (mean ± std, 3 seeds)")
print("="*50)
for name in OPTS:
    print(f"  {name:<10}  {np.mean(results[name]):.2f}% ± {np.std(results[name]):.2f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
colors = {'SGD-M':'#95A5A6','Adam':'#E74C3C','AdamW':'#F39C12','GONO':'#3498DB'}
fig, ax = plt.subplots(figsize=(7, 5))
means = [np.mean(results[n]) for n in OPTS]
stds  = [np.std(results[n])  for n in OPTS]
bars  = ax.bar(OPTS, means, color=[colors[n] for n in OPTS],
               yerr=stds, capsize=6, alpha=0.85, edgecolor='k')
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x()+bar.get_width()/2, m+s+0.1,
            f'{m:.2f}%', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Test Accuracy (%)'); ax.grid(True, alpha=0.3, axis='y')
ax.set_title('CIFAR-10 MLP (20 epochs, 10k subset, 3 seeds)')
plt.tight_layout()
plt.savefig('./results/exp4_cifar10.png', dpi=150, bbox_inches='tight')
print("\nSaved: ./results/exp4_cifar10.png")
