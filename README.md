# GONO: Gradient-Oriented Norm-Adaptive Optimizer

Official implementation of:

**"Directional Consistency as a Complementary Optimization Signal: The GONO Framework"**
Victor Daniel Gera — Anurag University
[[arXiv]](ARXIV_LINK_HERE) | [[Paper PDF]](PAPER_PDF_LINK_HERE)

---

## Key Idea

We identify the **direction-loss decoupling phenomenon**: an optimizer can exhibit
near-perfect directional consistency (cc_t → 1) while the loss barely decreases.

GONO exploits this by adapting Adam's β₁ using consecutive gradient cosine similarity:

```
cc_t  = <∇L_t, ∇L_{t-1}> / (||∇L_t|| · ||∇L_{t-1}||)   ← one dot product, O(d)
β₁,t  = clip(β₁ · (1 + λ · cc_t),  β₁_min,  β₁_max)
```

| cc_t | Meaning | GONO response |
|---|---|---|
| ≈ +1 | Gradients agree (smooth descent) | Increase β₁ → amplify momentum |
| ≈ 0  | Neutral | β₁ unchanged → identical to Adam |
| ≈ -1 | Gradients conflict (oscillation) | Decrease β₁ → suppress oscillation |

**Key result:** cc_t achieves **F1 = 1.00** for oscillation detection vs **F1 = 0.45** for gradient norm.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/gono-optimizer
cd gono-optimizer
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 2.0, torchvision, numpy, matplotlib, scikit-learn

---

## Quick Start

```python
from gono import GONO

# Drop-in replacement for Adam — same hyperparameters
optimizer = GONO(
    model.parameters(),
    lr=1e-3,
    beta1=0.9,
    lam=0.4,        # sensitivity to cc_t signal
    beta1_min=0.5,
    beta1_max=0.99
)

for x, y in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

---

## Reproducing Paper Results

**All experiments must be run from the repository root directory.**

```bash
cd gono-optimizer   # make sure you are here
```

### Experiment 1: Direction-Loss Decoupling Phenomenon
```bash
python experiments/exp1_decoupling.py
```
- Runtime: ~10 seconds (CPU only, no GPU needed)
- Output: `results/exp1_decoupling.png`
- Expected: cc_t stabilizes well before loss converges

### Experiment 2A: Oscillation Detection (Table 1 in paper)
```bash
python experiments/exp2a_oscillation_detection.py
```
- Runtime: ~5 seconds (CPU only)
- Output: `results/exp2a_oscillation_detection.png`
- Expected: Cons. Cosine F1=1.00, Gradient Norm F1=0.45

### Experiment 3: MNIST Benchmark (Table 2 in paper)
```bash
python experiments/exp3_mnist.py
```
- Runtime: ~15 minutes on GPU, ~2 hours on CPU
- Downloads MNIST automatically to `./data/`
- Output: `results/exp3_mnist.png`
- Expected: GONO ≈ 98.1%, Adam ≈ 97.1%, AdamW ≈ 98.2%

### Experiment 4: CIFAR-10 MLP Benchmark (Table 3 in paper)
```bash
python experiments/exp4_cifar10.py
```
- Runtime: ~10 minutes on GPU
- Downloads CIFAR-10 automatically to `./data/`
- Output: `results/exp4_cifar10.png`
- Expected: GONO ≈ 43.1%, AdamW ≈ 43.2%, Adam ≈ 42.8%

### Experiment 8: ResNet-18 Benchmark (Section 5.6 in paper)
```bash
python experiments/exp8_resnet18.py
```
- Runtime: ~2 hours on GPU (100 epochs × 3 seeds)
- Downloads CIFAR-10 automatically to `./data/`
- Output: `results/exp8_resnet18.png`
- Expected: GONO ≈ 75.4%, AdamW ≈ 76.9%, SGD-M ≈ 66.2%

---

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `lr` | 1e-3 | Learning rate (same as Adam) |
| `beta1` | 0.9 | Base momentum coefficient |
| `beta2` | 0.999 | Second moment coefficient |
| `eps` | 1e-8 | Numerical stability |
| `lam` | 0.4 | Sensitivity to cc_t signal |
| `beta1_min` | 0.5 | Minimum allowed beta1 |
| `beta1_max` | 0.99 | Maximum allowed beta1 |
| `weight_decay` | 0.0 | L2 regularization |

---

## Citation

```bibtex
@article{gera2026gono,
  title   = {Directional Consistency as a Complementary Optimization Signal:
             The GONO Framework},
  author  = {Gera, Victor Daniel},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {ARXIV_LINK_HERE}
}
```
