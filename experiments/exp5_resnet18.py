"""
Experiment 5: ResNet-18 on CIFAR-10 Benchmark

GONO vs AdamW vs SGD-M on full CIFAR-10 (50k train).
Architecture: ResNet-18, 100 epochs, cosine LR schedule.
Output: results/exp8_resnet18.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gono import GONO

os.makedirs('./results', exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── Data ──────────────────────────────────────────────────────────────────────
tf_tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                   T.ToTensor(),
                   T.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
tf_te = T.Compose([T.ToTensor(),
                   T.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
trl = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=tf_tr),
    batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
tel = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=tf_te),
    batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# ── Train ─────────────────────────────────────────────────────────────────────
def train(opt_name, epochs=100, seed=0):
    torch.manual_seed(seed)
    model = models.resnet18(weights=None, num_classes=10).to(DEVICE)
    if opt_name == 'AdamW':
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    elif opt_name == 'SGD-M':
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                              weight_decay=5e-4, nesterov=True)
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
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            model.eval()
            c = tot = 0
            with torch.no_grad():
                for x, y in tel:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    c += model(x).argmax(1).eq(y).sum().item(); tot += y.size(0)
            acc = 100. * c / tot
            accs.append((ep+1, acc))
            print(f"  [{opt_name}] ep{ep+1:3d}  acc={acc:.2f}%", flush=True)
    return accs

# ── Run ───────────────────────────────────────────────────────────────────────
OPTS   = ['AdamW', 'SGD-M', 'GONO']
SEEDS  = [0, 1, 2]
EPOCHS = 100

results = {}
for opt_name in OPTS:
    seed_finals = []
    for seed in SEEDS:
        print(f"\n{opt_name} seed={seed}")
        accs = train(opt_name, EPOCHS, seed)
        seed_finals.append(accs[-1][1])
    results[opt_name] = seed_finals

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("ResNet-18/CIFAR-10 RESULTS (mean ± std, 3 seeds)")
print("="*50)
for name in OPTS:
    print(f"  {name:<10}  {np.mean(results[name]):.2f}% ± {np.std(results[name]):.2f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
colors = {'AdamW':'#F39C12','SGD-M':'#95A5A6','GONO':'#3498DB'}
fig, ax = plt.subplots(figsize=(7, 5))
names  = list(results.keys())
means  = [np.mean(results[n]) for n in names]
stds   = [np.std(results[n])  for n in names]
bars   = ax.bar(names, means, color=[colors[n] for n in names],
                yerr=stds, capsize=6, alpha=0.85, edgecolor='k')
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x()+bar.get_width()/2, m+s+0.2,
            f'{m:.2f}%', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Test Accuracy (%)'); ax.grid(True, alpha=0.3, axis='y')
ax.set_title('ResNet-18 / CIFAR-10\n(100 epochs, 3 seeds)')
plt.tight_layout()
plt.savefig('./results/exp8_resnet18.png', dpi=150, bbox_inches='tight')
print("\nSaved: ./results/exp8_resnet18.png")
