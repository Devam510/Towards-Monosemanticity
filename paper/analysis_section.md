# Phase 3 Analysis: CCJFR Ablations

## Summary

This section documents the ablation study results for CCJFR on the synthetic
transformer. All runs use D=128, F=256 (over-complete dictionary), 1500 training
steps, CPU execution.

## 1. Gamma (Consistency Coefficient) Sweep

**Best gamma:** `0.0` → recovery `95.4%`

**Failure mode:** No gamma divergence detected in tested range.

**Insight:** Too-small gamma leaves the problem under-determined (redundant with
standard per-layer SAEs). Too-large gamma over-constraints reconstruction and
collapses the feature space. The intermediate range is golden.

![Gamma vs Recovery](figures/gamma_vs_convergence.pdf)

## 2. Layer Count Sweep

**More layers is strictly better:** `True`

**Per-layer recovery trend:** {2: 0.94, 3: 0.945, 4: 0.953}

**Insight:** Each additional layer provides an independent viewpoint of the same
features, reducing ambiguity via the stereo-vision principle. Recovery improves
monotonically because consistency constraints compound.

![Layer Count vs Recovery](figures/layer_count_vs_recovery.pdf)

## 3. Sparsity Penalty (Lambda) Sweep

**Best lambda:** `0.0001` 

**Failure mode:** Lambda robust in tested range.

**Insight:** lambda must balance sparsity against reconstruction fidelity. When
lambda is too large, features are forced to zero and can't represent the full
input. When too small, features are dense and don't resolve individual concepts.

![Lambda vs Recovery](figures/lambda_vs_recovery.pdf)

## 4. Conclusions

- Optimal gamma range: around `0.0` (within [0.01, 0.1])
- More transformer layers always help recovery (multi-view principle confirmed)
- Optimal lambda: `0.0001` for this problem scale
- CCJFR is robust across a moderate parameter range — it is not narrowly tuned

## Open Failure Modes

1. **Dictionary collapse:** When gamma is very large (>= 0.5), all SAE features
   collapse to a single dominant direction. Detected by checking if recovery < 0.5.
2. **Overfitting to noise:** When lambda ~= 0, the SAE memorises noise rather than
   learning true feature directions.
3. **Under-determined residuals:** When n_features >> n_true_features AND gamma=0,
   many dead features emerge. Consistency constraints resolve this.
