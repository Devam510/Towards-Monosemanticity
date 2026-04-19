"""
CCJFR: Computation-Constrained Joint Feature Recovery

The key innovation of this paper.

Standard SAEs train per-layer independently. This is under-determined:
many different decompositions reconstruct equally well.

CCJFR adds a computation consistency constraint:
  Features at layer l+1 must be derivable from features at layer l
  by running them through the model's ACTUAL computation.

This turns the problem from under-determined to (potentially) uniquely
solvable — like stereo vision, where multiple camera views make 3D 
reconstruction unique.

Additionally, boundary anchoring (also novel) provides:
  - Layer 0 features anchored to known embedding directions
  - Last layer features anchored to unembedding directions

Loss:
  L = Σ_l [ MSE(h_l, x_hat_l) + λ·||z_l||_1 ]        # Reconstruction + sparsity
    + γ · Σ_l MSE(z_{l+1}, Enc_{l+1}(T_l(Dec_l(z_l))))  # Computation consistency
    + α · L_embed                                         # Boundary: embedding
    + β · L_unembed                                       # Boundary: unembedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Callable
from .sae import SparseAutoencoder, SAEConfig


@dataclass
class CCJFRConfig:
    n_layers: int = 3                   # Number of layers to jointly train
    d_model: int = 256                  # Residual stream dimension
    n_features: int = 1024             # Dictionary size per layer
    k: int = 0                         # TopK sparsity (0 = L1 mode)
    l1_coeff: float = 8e-5             # Sparsity penalty (for L1 mode)
    consist_coeff: float = 0.1         # γ — consistency penalty
    embed_anchor_coeff: float = 0.1    # α — embedding boundary penalty
    unembed_anchor_coeff: float = 0.1  # β — unembedding boundary penalty
    lr: float = 2e-4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Annealing: consistency weight ramps up from 0 to consist_coeff
    consist_anneal_steps: int = 5000


class CCJFR(nn.Module):
    """
    Computation-Constrained Joint Feature Recovery.

    Contains one SAE per transformer layer, trained jointly with
    consistency constraints between adjacent layers.
    """

    def __init__(
        self,
        config: CCJFRConfig,
        # Optional: known boundary feature directions
        embed_matrix: Optional[torch.Tensor] = None,    # [D, V] embedding E
        unembed_matrix: Optional[torch.Tensor] = None,  # [V, D] unembedding U
    ):
        super().__init__()
        self.config = config
        torch.manual_seed(config.seed)

        # One SAE per layer
        sae_config = SAEConfig(
            d_model=config.d_model,
            n_features=config.n_features,
            k=config.k,
            l1_coeff=config.l1_coeff,
            lr=config.lr,
        )
        self.saes = nn.ModuleList([
            SparseAutoencoder(sae_config) for _ in range(config.n_layers)
        ])

        # Store boundary matrices as buffers (not trained)
        if embed_matrix is not None:
            self.register_buffer('embed_matrix', embed_matrix)
        else:
            self.embed_matrix = None

        if unembed_matrix is not None:
            self.register_buffer('unembed_matrix', unembed_matrix)
        else:
            self.unembed_matrix = None

        # Annealing step counter
        self.register_buffer('_anneal_step', torch.tensor(0))

    @property
    def current_consist_coeff(self) -> float:
        """Returns annealed consistency coefficient."""
        t = self._anneal_step.item()
        T = self.config.consist_anneal_steps
        frac = min(1.0, t / max(T, 1))
        return self.config.consist_coeff * frac

    def _recon_sparsity_loss(
        self, h: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard SAE reconstruction + sparsity loss at one layer."""
        sae = self.saes[layer_idx]
        z = sae.encode(h)
        h_hat = sae.decode(z)
        recon = ((h - h_hat) ** 2).mean()
        sparse = self.config.l1_coeff * z.abs().mean()
        return recon + sparse, z, h_hat

    def _consistency_loss(
        self,
        z_l: torch.Tensor,     # [batch, F] features at layer l
        h_next: torch.Tensor,  # [batch, D] actual activations at layer l+1
        layer_idx: int,        # l
        compute_fn: Callable   # T_l: h_l -> h_{l+1} (actual transformer computation)
    ) -> torch.Tensor:
        """
        Consistency loss between layer l and l+1.

        Checks: Enc_{l+1}(T_l(Dec_l(z_l))) ≈ Enc_{l+1}(h_{l+1})

        i.e., if you take the layer-l features, decode to h-space,
        run through the actual transformer computation, then re-encode —
        you should get the same features as directly encoding h_{l+1}.
        """
        sae_l = self.saes[layer_idx]
        sae_next = self.saes[layer_idx + 1]

        # Decode layer-l features back to activation space
        h_l_hat = sae_l.decode(z_l)                   # [batch, D]

        # Run through actual computation (no gradient through T_l — we're
        # using it as a fixed "oracle" of the model's computation)
        with torch.no_grad():
            h_next_predicted = compute_fn(h_l_hat)    # [batch, D]

        # Encode predicted h_{l+1} with layer-(l+1) SAE
        z_next_predicted = sae_next.encode(h_next_predicted)  # [batch, F]

        # Encode actual h_{l+1}
        z_next_actual = sae_next.encode(h_next.detach())      # [batch, F]

        # Loss: these should match
        return ((z_next_predicted - z_next_actual) ** 2).mean()

    def _embedding_anchor_loss(self) -> torch.Tensor:
        """
        Anchor layer-0 SAE decoder directions to embedding columns.

        Strong embeddings should be recovered as strong SAE features.
        If embed_matrix is [D, V], we want SAE features to span this space.

        Loss: minimize the distance from each embedding direction to
              the nearest SAE feature direction.
        """
        if self.embed_matrix is None:
            return torch.tensor(0.0)

        sae_dirs = self.saes[0].feature_directions()  # [F, D]
        # If embed_matrix is [V, D]
        embed_dirs = F.normalize(self.embed_matrix, dim=1)  # [V, D]

        # For each embedding direction, find closest SAE feature
        sim = embed_dirs @ F.normalize(sae_dirs, dim=1).T  # [V, D] @ [D, F] = [V, F]
        max_sim, _ = sim.max(dim=1)  # [V]

        # We want max similarity to be high → loss is 1 - mean(max_sim)
        return (1 - max_sim).mean()

    def _unembedding_anchor_loss(
        self,
        z_last: torch.Tensor,  # [batch, F] features at last layer
        actual_logits: torch.Tensor  # [batch, V] actual model logits
    ) -> torch.Tensor:
        """
        Anchor last-layer features to unembedding matrix.

        If unembed_matrix is [V, D], then the predicted logit effect of
        a feature f with decoder direction d_f is: W_U @ d_f

        We check: does the SAE's reconstruction of the last layer,
        projected through W_U, match the actual logits?

        Loss: MSE between predicted logits and actual logits.
        """
        if self.unembed_matrix is None:
            return torch.tensor(0.0)

        sae_last = self.saes[-1]
        h_last_hat = sae_last.decode(z_last)  # [batch, D]

        # Predicted logits via unembedding
        predicted_logits = h_last_hat @ self.unembed_matrix.T  # [batch, V]

        return ((predicted_logits - actual_logits.detach()) ** 2).mean()

    def forward(
        self,
        activations: list[torch.Tensor],  # [h_0, h_1, ... h_L] each [batch, D]
        compute_fns: Optional[list[Callable]] = None,  # T_0, T_1, ... T_{L-1}
        actual_logits: Optional[torch.Tensor] = None,  # [batch, V] for unembed anchor
        consist_coeff_override: Optional[float] = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Full CCJFR forward pass.

        Args:
            activations: list of layer activations [h_0, ..., h_L]
            compute_fns: list of T_l functions (transformer computation per layer)
            actual_logits: model output logits (for unembedding anchor)
            consist_coeff_override: override annealed coefficient (for testing)

        Returns:
            total_loss: scalar
            metrics: dict with component losses and feature activations
        """
        assert len(activations) == self.config.n_layers, \
            f"Expected {self.config.n_layers} activations, got {len(activations)}"

        γ = consist_coeff_override if consist_coeff_override is not None else self.current_consist_coeff
        α = self.config.embed_anchor_coeff
        β = self.config.unembed_anchor_coeff

        total_loss = torch.tensor(0.0, device=activations[0].device)
        feature_acts = []

        # --- Per-layer reconstruction + sparsity ---
        recon_loss_total = 0.0
        for l, h in enumerate(activations):
            loss_l, z_l, _ = self._recon_sparsity_loss(h, l)
            total_loss = total_loss + loss_l
            recon_loss_total += loss_l.item()
            feature_acts.append(z_l)

        # --- Consistency losses ---
        consist_loss_total = 0.0
        if compute_fns is not None and γ > 0:
            assert len(compute_fns) == self.config.n_layers - 1
            for l in range(self.config.n_layers - 1):
                c_loss = self._consistency_loss(
                    z_l=feature_acts[l],
                    h_next=activations[l + 1],
                    layer_idx=l,
                    compute_fn=compute_fns[l]
                )
                total_loss = total_loss + γ * c_loss
                consist_loss_total += c_loss.item()

        # --- Boundary anchoring ---
        embed_loss = self._embedding_anchor_loss() if α > 0 else torch.tensor(0.0)
        total_loss = total_loss + α * embed_loss

        unembed_loss = torch.tensor(0.0)
        if actual_logits is not None and β > 0:
            unembed_loss = self._unembedding_anchor_loss(feature_acts[-1], actual_logits)
            total_loss = total_loss + β * unembed_loss

        # Increment annealing step
        self._anneal_step += 1

        metrics = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss_total,
            "consist_loss": consist_loss_total,
            "embed_anchor_loss": embed_loss.item() if isinstance(embed_loss, torch.Tensor) else embed_loss,
            "unembed_anchor_loss": unembed_loss.item() if isinstance(unembed_loss, torch.Tensor) else unembed_loss,
            "consist_coeff": γ,
            "feature_acts": [z.detach() for z in feature_acts],
        }

        return total_loss, metrics

    def feature_directions_at(self, layer: int) -> torch.Tensor:
        """Return [F, D] feature directions at a given layer."""
        return self.saes[layer].feature_directions()

    def normalize_all_decoders(self):
        """Normalize decoder columns for all SAEs after optimizer step."""
        for sae in self.saes:
            sae._normalize_decoder()
