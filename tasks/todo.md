# CCJFR Research Implementation & Anthropic Submission Tracker

## Previous Phases (Completed)
- `[x]` **Phase 1:** Synthetic Proof-of-Concept (CCJFR convergence 0.988)
- `[x]` **Phase 2:** Real Model Validation (Baseline scripts, CCJFR integration)
- `[x]` **Phase 3:** Ablations + Analysis (Layer count scaling, gamma stability)
- `[x]` **Phase 4:** Initial Write-up (Drafts of LaTeX and Alignment Forum post)

---

## Phase A: GPT-2 Small Infrastructure
- `[x]` Modify `scripts/cache_activations.py` to support `--model gpt2`, `--seq_len 64`, `--fp16`, `--n_batches 500` (Implemented via a dedicated 13_cache_gpt2_activations.py)
- `[x]` Create `src/models/gpt2_wrapper.py` (TransformerLens wrapper exposing W_E and W_U)
- `[x]` Create `experiments/13_cache_gpt2_activations.py` (Driver script)
- `[ ]` Run `13_cache_gpt2_activations.py` and verify `experiments/cache/gpt2_small/` is populated

## Phase B: Verified Real-Model Experiments
- `[x]` Create `experiments/14_gpt2_baseline_saes.py` (Train standard SAE on GPT-2 layer 0, 2 seeds, 5000 steps)
- `[x]` Create `experiments/15_gpt2_ccjfr.py` (Train CCJFR on GPT-2 layers 0-3 jointly, 2 seeds, 5000 steps)
- `[ ]` Fix `experiments/09_causal_evaluation.py` to use trained SAE weights, not random weights
- `[ ]` Update `experiments/10_seven_evidence_lines.py` to use trained weights and output real numbers

## Phase C: Published SAE Comparison
- `[x]` Create `experiments/16_compare_published_saes.py`
- `[ ]` Download `jbloom/GPT2-Small-SAEs` weights from HuggingFace
- `[ ]` Measure and compare cross-seed convergence with CCJFR

## Phase D: Full Identifiability Theorem
- `[x]` Add Lemma 1 (Standard SAE rotational invariance) to `paper/appendix.tex`
- `[x]` Add Lemma 2 (Consistency loss constraint on rotation) to `paper/appendix.tex`
- `[x]` Add Lemma 3 (Boundary anchoring uniqueness) to `paper/appendix.tex`
- `[x]` Write Theorem 1 (CCJFR Identifiability Proof) in `paper/appendix.tex`
- `[x]` Add Proposition 1 to `paper/main.tex` (Informal theorem)

## Phase E: Paper Polish & Results Integration
- `[ ]` Fill `\TODO{}` in `paper/main.tex` with real Phase B and C numbers
- `[ ]` Add Table 3 comparing CCJFR vs Published SAE on GPT-2
- `[ ]` Add `gpt2_cross_seed_comparison.pdf` chart
- `[ ]` Run `pytest tests/ -v` to ensure 22/22 tests are still passing

### ⛔ ANTHROPIC SUBMISSION GATE
- `[ ]` GPT-2 Small CCJFR convergence > standard SAE convergence (p < 0.05)
- `[ ]` GPT-2 Small CCJFR convergence > published `jbloom` SAE convergence
- `[ ]` Causal KL: CCJFR > baseline (p < 0.05)
- `[ ]` Mathematical proof completed in Appendix
- `[ ]` Readme and final paper ready
