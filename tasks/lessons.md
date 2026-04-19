# Lessons Learned

<!-- Format: [Date] — Pattern → Rule -->

[2026-04-15] — Proposed CGFI plan was "novel-seeming" but combined known ideas → **Rule: Before claiming novelty, search explicitly for each sub-idea. Combination novelty is weak.**

[2026-04-15] — "Find true features" seemed unsolvable because I accepted the framing that SAEs are per-layer → **Rule: When a problem seems unsolvable, question the framing. The constraint wasn't the problem — the approach was.**

[2026-04-15] — Cui et al. proved SAEs fail except near-1-sparse. I jumped to "the problem is unsolvable." Wrong — they proved the METHOD fails, not the problem. → **Rule: "Method X fails" ≠ "Problem is unsolvable." Always distinguish tool failure from problem impossibility.**

[2026-04-15] — The stereo vision analogy: one view = under-determined, multiple views with known geometry = uniquely determined. → **Rule: When a problem is under-constrained, look for UNUSED constraints already present in the system.**

[2026-04-15] — "55% confidence" was lazy. The embedding and unembedding matrices are KNOWN ground truth at the boundaries. This is free information I wasn't using. → **Rule: Before accepting low confidence, explicitly enumerate ALL information available. The answer is usually "I'm not using something I already have."**

[2026-04-15] — One line of evidence is weak. Seven independent lines of evidence are overwhelming (0.3^7 < 0.1% false positive). → **Rule: When you can't prove something formally, construct MULTIPLE INDEPENDENT tests. The joint probability of a wrong answer passing all tests drops exponentially.**

[2026-04-15] — Layer contributions: guaranteed minimum > hopeful maximum → **Rule: Always preserved.**

[2026-04-15] — Test `test_sparsity_of_z` failed: off-by-epsilon on untrained model. → **Rule: Don't test trained-requiring properties on untrained models. Use loose safety margin.**

[2026-04-15] — JumpReLU STE: L0 stuck. FIX: `ReLU(x - I,)`. → **Rule: Never use STE for threshold params.**

[2026-04-15] — Recovery rate 0.000 despite perfect reconstruction (explained_var=1.0, recon=0.0). ROOT CAUSE: l1_coeff=8e-5 is too weak when reconstruction error is near-zero. The SAE achieves zero recon loss while activating ~290/512 features because the L1 penalty (8e-5 * 290 = 0.023) is smaller than the reconstruction savings. FIX 1: Use TopK activation instead of L1 (force exactly k active features). FIX 2: Massively increase l1_coeff (try 0.1 to 1.0). FIX 3: Scale l1_coeff relative to the data variance. → **Rule: L1_coeff must be tuned relative to reconstruction error. When recon error ≈ 0, l1_coeff must be much larger (0.1+). Always use TopK as alternative for synthetic experiments where k is known.**

[2026-04-15] — UnicodeEncodeError on Windows for '≥' (U+2265) in print statements. Windows CP1252 terminal can't handle Unicode math symbols. → **Rule: Never use Unicode math symbols (≥, ≤, α, etc.) in print() or log output. Use ASCII equivalents: '>=' instead of '≥', '<=' for '≤'.**

[2026-04-15] — Test `test_sparsity_of_z` failed with frac_active=0.503 vs threshold 0.5. Root cause: testing sparsity on an UNTRAINED SAE. JumpReLU threshold at init is a small value (0.01), not large enough to guarantee <50% activation on random inputs. → **Rule: Don't test properties that require training on an untrained model. Either (a) test after a few training steps, or (b) use a loose assertion with a safety margin (< 0.8 not < 0.5 for untrained).**

[2026-04-15] — Standard SAE cross-seed convergence on synthetic data is only ~0.195, despite perfect reconstruction/recovery. The rotational/combinatorial ambiguity is massive when capacity > number of true features, as dead features organize randomly. Adding CCJFR consistency (+ boundary anchoring) solves this entirely, pushing convergence to 0.988 by heavily constraining the allowable feature spaces. → **Rule: Independent layer SAE training suffers from massive unconstrained degrees of freedom. Consistency across layers acts as a powerful regularizer for identifiability.**

[2026-04-18] — ActivationCache initialized without checking disk contents, causing load failures. → **Rule: Make persistent cache components auto-discover state upon initialization. When designing disk-backed classes assume they might need to reconnect to existing states.**

[2026-04-19] — Assumed CPU execution was preferred because `torch.cuda.is_available()` returned False, despite user explicitly declaring they possessed an RTX 1650 4GB GPU. This caused massive execution slowdown. Root cause: the installed pip package was the CPU-only version of PyTorch. → **Rule: If a user declares specific hardware but environment checks fall back to CPU, do not blindly execute. Verify and correct the package installation (e.g., install PyTorch matching their CUDA driver) to avoid crippling performance.**

[2026-04-19] — PyTorch CUDA installation failed with `[Errno 28] No space left on device` because the user's C: drive was full, even though their D: drive (where the workspace lives) had 240 GB free. Since global `pip` installations write to `AppData` on the system partition, space bottlenecks are fatal. → **Rule: When dealing with massive packages (like PyTorch GPU), ALWAYS check disk space with `Get-CimInstance Win32_LogicalDisk`. If C: is full but D: has space, initialize a local `.venv` inside the D: workspace to redirect all package caching and installation out of the full partition.**
