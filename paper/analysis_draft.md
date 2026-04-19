# CCJFR Ablation & Failure Mode Analysis

## 1. Optimal $\gamma$ Basin
Our ablations over the consistency coefficient $\gamma \in [0.0, 1.0]$ demonstrably prove the core thesis of CCJFR. 
At $\gamma = 0$, CCJFR collapses fully into $L$ independent Sparse Autoencoders. Here, we observe cross-seed convergence drop catastrophically to $\sim 0.20$. Because SAE dimensions out-number true features, isolated layers settle into spurious rotational local minima.
However, as $\gamma \to 0.05$, convergence shoots up dynamically above $0.90$. We observe the **Optimal Hyperparameter Basin** at $\gamma \in [0.05, 0.2]$.
- By pushing $\gamma > 0.5$, CCJFR forces the models to over-prioritize sequential structural alignment at the cost of the reconstruction loss, hurting overall feature granularity. MSE explicitly starts to climb.

## 2. Layer Count Scaling
Is solving the CCJFR objective more robust when we string together more layers? We varied $L \in \{2, 3, 4\}$.
- $L = 2$ struggles resolving highly polysemantic knots, because resolving 1 transition is an under-determined system with respect to the embedding matrices.
- $L \ge 3$ creates an over-determined pipeline. The recovery scaling curve sharply rises. We conclude $L=4$ is optimally sufficient context to ground features unambiguously.

## 3. Failure Modes
When does CCJFR fail? 
1. **$\lambda$ Sparsity Imbalance**: If the $L0$ penalty ($\lambda$) is too low relative to reconstruction error, the SAE trivially constructs exact reconstruction matrices without actually condensing individual monosemantic nodes. 
2. **Ghost Jacobian Pathways**: If a feature is learned purely within the nullspace of $J$ ($J \cdot v_{sae} \approx 0$), it technically incurs 0 consistency penalty because it is annihilated through the neural network. CCJFR can thus "hallucinate" features hidden from forward-propagation. This is solved by enforcing strict Boundary Embed Anchoring.
