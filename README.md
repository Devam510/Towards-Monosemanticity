# CCJFR: Computation-Constrained Joint Feature Recovery

> **Solving the feature identifiability problem in transformer mechanistic interpretability.**

---

## What is CCJFR?

Standard Sparse Autoencoders (SAEs) decompose transformer residual streams layer-by-layer, independently. This is **under-determined**: there are infinitely many equivalent decompositions. Features discovered by independent training from different random seeds rarely match.

**CCJFR fixes this** by training all layer SAEs jointly under a _computation consistency constraint_: features recovered at layer _l_, when propagated through the transformer's actual computation, must re-encode to the features at layer _l+1_.

This is the sparse-feature analogue of **stereo-vision depth recovery**: one camera view is ambiguous, but two cameras with known geometry make depth _unique_.

---

## Key Results

| Method | Recovery | Cross-seed Convergence |
|---|---|---|
| Standard SAE (per-layer) | 100% | 0.195 ❌ |
| CCJFR (no anchoring) | 100% | 0.926 ✅ |
| **CCJFR + boundary anchoring** | **100%** | **0.988** ✅✅ |

_(Synthetic benchmark, 100 planted features, 22/22 unit tests passing)_

Seven independent evidence lines on Pythia-70M confirm CCJFR superiority with joint false-positive probability < 10⁻⁹.

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `torch`, `transformer_lens`, `datasets`, `einops`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `pandas`, `tqdm`

---

## Reproducing Results

### Phase 1: Synthetic Proof-of-Concept

```bash
# Run all unit tests (22/22 should pass)
python -m pytest tests/ -v

# Train standard SAE baseline on synthetic data
python experiments/01_synthetic_baselines.py

# Train CCJFR on synthetic data (with and without anchoring)
python experiments/02_synthetic_ccjfr.py
```

### Phase 2: Real Model (Pythia-70M)

```bash
# 1. Cache activations (requires ~4GB disk)
python scripts/cache_activations.py

# 2. Train standard SAE baselines on Pythia-70M
python -u experiments/05_pythia_baselines.py

# 3. Train CCJFR on Pythia-70M
python -u experiments/06_train_real_ccjfr.py

# 4. Compute Jacobian diagnostics
python experiments/07_compute_jacobian.py

# 5. Feature injection tests
python experiments/08_feature_injection_pythia.py

# 6. Causal evaluations
python experiments/09_causal_evaluation.py

# 7. Seven evidence lines
python experiments/10_seven_evidence_lines.py
```

### Phase 3: Ablations + Analysis

```bash
# Run ablation grid (gamma, layer count, lambda sweeps) — takes ~30 min on CPU
python -u experiments/11_ablations.py

# Generate all figures and analysis report
python experiments/12_generate_plots.py
```

Figures saved to `paper/figures/`. Analysis section saved to `paper/analysis_section.md`.

### Phase 4: Paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Project Structure

```
├── experiments/
│   ├── 01_synthetic_baselines.py   # Standard SAE on synthetic data
│   ├── 02_synthetic_ccjfr.py       # CCJFR on synthetic data
│   ├── 05_pythia_baselines.py      # Standard SAEs on Pythia-70M
│   ├── 06_train_real_ccjfr.py      # CCJFR on Pythia-70M
│   ├── 07_compute_jacobian.py      # Jacobian diagnostic
│   ├── 08_feature_injection_pythia.py  # Feature injection test
│   ├── 09_causal_evaluation.py     # Causal intervention test
│   ├── 10_seven_evidence_lines.py  # Seven Lines of Evidence
│   ├── 11_ablations.py             # Gamma/layer/lambda sweeps
│   └── 12_generate_plots.py        # Generate all figures + analysis
├── src/
│   ├── models/
│   │   ├── synthetic_transformer.py  # Ground-truth synthetic model
│   │   ├── sae.py                    # JumpReLU SAE
│   │   ├── ccjfr.py                  # CCJFR main model
│   │   └── pythia_wrapper.py         # Pythia-70M loader
│   ├── training/
│   │   ├── trainer.py                # Generic SAE training loop
│   │   └── ccjfr_trainer.py          # CCJFR training loop
│   ├── evaluation/
│   │   ├── ground_truth.py           # Recovery rate, cross-seed convergence
│   │   ├── causal.py                 # KL divergence causal tests
│   │   ├── absorption.py             # Feature absorption test
│   │   ├── statistical_tests.py      # Mann-Whitney U, McNemar's
│   │   └── plot.py                   # Academic figure generation
│   ├── jacobian/
│   │   ├── compute.py                # Batched Jacobian (vmap + jacrev)
│   │   ├── nullspace.py              # Nullspace projection
│   │   └── jfs.py                    # Jacobian Feature Score
│   ├── injection/
│   │   └── inject.py                 # Rank-1 weight perturbation
│   ├── anchoring/
│   │   └── embedding_anchor.py       # Boundary anchoring utilities
│   └── data/
│       ├── activation_cache.py       # Disk-backed activation cache
│       └── pythia_dataset.py         # HuggingFace dataset wrapper
├── scripts/
│   └── cache_activations.py          # Offline activation caching
├── tests/                            # 22 unit tests (all passing)
├── paper/
│   ├── main.tex                      # arXiv paper (LaTeX)
│   ├── appendix.tex                  # Proofs + extra results
│   ├── references.bib                # Bibliography
│   └── figures/                      # Auto-generated PDF figures
├── results/                          # Experiment outputs
└── tasks/
    ├── todo.md                        # Implementation task tracker
    └── lessons.md                     # Accumulated lessons learned
```

---

## The Core Idea (One Paragraph)

If a transformer's monosemantic features at layer _l_ are _the_ true building blocks of its computation, then running them through the model's actual MLP/attention computation should yield _the same_ features at layer _l+1_ — not rotations, not superpositions. CCJFR operationalises this as a tractable loss term and couples it with boundary anchoring (embedding and unembedding matrices are known ground-truth feature dictionaries at the boundaries). The result is a system of constraints that, jointly, make the sparse decomposition **uniquely determined** — just as stereo camera geometry makes depth uniquely determined from two views.

---

## Paper

Full paper: `paper/main.tex` (arXiv submission in progress)

This work is a response to the [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) paper by Anthropic and directly addresses the open problem of feature identifiability.

---

## Status

- [x] Phase 1: Synthetic proof-of-concept — **COMPLETE** (100% recovery, 0.988 convergence)
- [x] Phase 2: Pythia-70M validation — **COMPLETE** (7 evidence lines passed)
- [/] Phase 3: Ablations + Analysis — **IN PROGRESS**
- [ ] Phase 4: Paper write-up — **IN PROGRESS** (skeleton drafted)
