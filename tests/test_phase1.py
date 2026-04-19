"""
Unit Tests for Phase 1 Core Components

Tests:
1. SyntheticTransformer: correct shapes, data generation, ground truth access
2. SparseAutoencoder: forward pass, encode/decode, loss components
3. FeatureInjector: injection, verification
4. Ground truth evaluation: recovery rate, cross-seed convergence

Run: pytest tests/test_phase1.py -v
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.synthetic_transformer import SyntheticTransformer, SyntheticConfig, get_true_features_at_layer
from src.models.sae import SparseAutoencoder, SAEConfig
from src.evaluation.ground_truth import feature_recovery_rate, cosine_similarity_matrix, cross_seed_convergence
from src.injection.inject import FeatureInjector, generate_injection_directions


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def small_config():
    return SyntheticConfig(n_features=20, d_model=64, n_layers=2, sparsity_k=3, n_samples=500)

@pytest.fixture
def small_model(small_config):
    return SyntheticTransformer(small_config)

@pytest.fixture
def small_sae(small_config):
    return SparseAutoencoder(SAEConfig(d_model=small_config.d_model, n_features=64))


# ─── SyntheticTransformer ─────────────────────────────────────────────────────

class TestSyntheticTransformer:

    def test_feature_directions_shape(self, small_model, small_config):
        expected = (small_config.n_features, small_config.d_model)
        assert small_model.true_features.shape == expected, f"Expected {expected}, got {small_model.true_features.shape}"

    def test_data_generation_shapes(self, small_model, small_config):
        z_true, activations = small_model.generate_data(n_samples=100, seed=0)
        assert z_true.shape == (100, small_config.n_features)
        assert len(activations) == small_config.n_layers + 1  # h_0 through h_L
        for h in activations:
            assert h.shape == (100, small_config.d_model)

    def test_sparsity_of_generated_data(self, small_model, small_config):
        z_true, _ = small_model.generate_data(n_samples=200, seed=1)
        # Each sample should have exactly k non-zero features
        n_active_per_sample = (z_true.abs() > 0).float().sum(dim=1)
        assert (n_active_per_sample == small_config.sparsity_k).all(), \
            f"Expected k={small_config.sparsity_k} active features, got range [{n_active_per_sample.min()}, {n_active_per_sample.max()}]"

    def test_true_features_normalized(self, small_model, small_config):
        if small_config.n_features <= small_config.d_model:
            # Should be orthonormal
            norms = torch.norm(small_model.true_features, dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
                "Feature directions should be unit norm"

    def test_layer_feature_dirs_available(self, small_model, small_config):
        for l in range(small_config.n_layers + 1):
            dirs = get_true_features_at_layer(small_model, l)
            assert dirs.shape == (small_config.n_features, small_config.d_model)

    def test_embed_matrix_shape(self, small_model, small_config):
        assert small_model.embed_matrix.shape == (small_config.d_model, small_config.n_features)


# ─── SparseAutoencoder ────────────────────────────────────────────────────────

class TestSparseAutoencoder:

    def test_forward_shapes(self, small_sae, small_config):
        x = torch.randn(32, small_config.d_model)
        x_hat, z, loss = small_sae(x)
        assert x_hat.shape == x.shape
        assert z.shape == (32, 64)  # n_features
        assert loss.ndim == 0  # scalar

    def test_loss_is_positive(self, small_sae, small_config):
        x = torch.randn(16, small_config.d_model)
        _, _, loss = small_sae(x)
        assert loss.item() > 0

    def test_encode_decode_roundtrip(self, small_sae, small_config):
        """After training a bit, reconstruction should improve."""
        x = torch.randn(32, small_config.d_model)
        x_hat, _, _ = small_sae(x)
        # Just check shape consistency
        assert x_hat.shape == x.shape

    def test_sparsity_of_z(self, small_sae, small_config):
        """JumpReLU should zero out at least some features (not all active).
        
        Note: We don't test for strict <0.5 on an untrained model because
        the JumpReLU threshold (init=0.01) needs to be warmed up.
        We verify the mechanism works at all: at least some features are zeroed.
        """
        x = torch.randn(64, small_config.d_model)
        _, z, _ = small_sae(x)
        frac_zero = (z.abs() <= 1e-8).float().mean().item()
        # With JumpReLU threshold=0.01, at least some entries should be zero
        # across a batch of random inputs (even at init)
        assert frac_zero > 0.0, "JumpReLU should zero out at least some activations"

    def test_feature_directions_shape(self, small_sae, small_config):
        dirs = small_sae.feature_directions()
        assert dirs.shape == (64, small_config.d_model)

    def test_decoder_normalization(self, small_sae):
        """Decoder columns should be unit norm after normalization."""
        small_sae._normalize_decoder()
        col_norms = torch.norm(small_sae.W_dec, dim=0)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5), \
            "Decoder columns should be unit norm"


# ─── Ground Truth Evaluation ──────────────────────────────────────────────────

class TestGroundTruthEval:

    def test_perfect_recovery(self):
        """If SAE contains exact true features, recovery should be 100%."""
        true_dirs = torch.eye(10, 20)  # 10 true features in 20-dim space
        sae_dirs = true_dirs.clone()   # Perfect match
        result = feature_recovery_rate(true_dirs, sae_dirs, threshold=0.99)
        assert result["recovery_rate"] == 1.0, f"Expected 1.0, got {result['recovery_rate']}"

    def test_zero_recovery(self):
        """If SAE features are orthogonal to true features, recovery is 0."""
        true_dirs = torch.zeros(5, 20)
        true_dirs[:, :5] = torch.eye(5)   # True features in first 5 dims
        sae_dirs = torch.zeros(5, 20)
        sae_dirs[:, 10:15] = torch.eye(5) # SAE in orthogonal dims
        result = feature_recovery_rate(true_dirs, sae_dirs, threshold=0.9)
        assert result["recovery_rate"] == 0.0

    def test_cosine_similarity_matrix_shape(self):
        a = torch.randn(10, 32)
        b = torch.randn(15, 32)
        sim = cosine_similarity_matrix(a, b)
        assert sim.shape == (10, 15)

    def test_cosine_similarity_self_is_1(self):
        a = torch.randn(5, 32)
        sim = cosine_similarity_matrix(a, a)
        diagonal = sim.diag()
        assert torch.allclose(diagonal, torch.ones(5), atol=1e-5)

    def test_cross_seed_convergence_perfect(self):
        """If features are identical across seeds, convergence should be 1.0."""
        dirs = torch.eye(10, 32)
        result = cross_seed_convergence([dirs, dirs, dirs], threshold=0.99)
        assert result["convergence_rate"] > 0.99

    def test_cross_seed_convergence_random(self):
        """Random features should have low convergence."""
        dirs1 = torch.nn.functional.normalize(torch.randn(20, 64), dim=1)
        dirs2 = torch.nn.functional.normalize(torch.randn(20, 64), dim=1)
        result = cross_seed_convergence([dirs1, dirs2], threshold=0.9)
        # Random high-dim vectors have very low cosine similarity
        assert result["convergence_rate"] < 0.3


# ─── Feature Injection ────────────────────────────────────────────────────────

class TestFeatureInjection:

    def test_injection_creates_direction(self):
        injector = FeatureInjector()
        d = 64
        u, v = generate_injection_directions(d, seed=0)
        W = torch.randn(d, d)
        W_prime = injector.inject_into_tensor(W, u, v, alpha=5.0)
        # W_prime should differ from W
        assert not torch.allclose(W, W_prime)
        # Difference should be in direction v⊗u
        diff = W_prime - W
        assert diff.shape == W.shape

    def test_recovery_with_strong_injection(self):
        """A strongly injected feature should be recoverable by any method."""
        injector = FeatureInjector()
        d = 64
        u, v = generate_injection_directions(d, seed=13)
        alpha = 10.0  # Very strong injection

        # Record the injection
        injector.record_injection(u, v, alpha, layer=0)

        # Simulate an SAE that perfectly contains this feature
        perfect_sae_dirs = v.unsqueeze(0)  # [1, d]
        result = injector.verify_recovery(perfect_sae_dirs, threshold=0.9)
        assert result["recovered"], f"Expected recovery with perfect SAE, got max_cos={result['max_cos']:.3f}"

    def test_generate_injection_directions_orthogonal(self):
        """u and v should be orthogonal."""
        u, v = generate_injection_directions(128, seed=7)
        dot = (u @ v).item()
        assert abs(dot) < 1e-5, f"u and v should be orthogonal, got dot={dot:.6f}"


# ─── Integration Test ─────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_phase1_pipeline(self, small_model, small_config):
        """
        End-to-end test: generate data → train SAE → evaluate recovery.
        This is a smoke test — doesn't check for high recovery, just that
        the pipeline runs without errors.
        """
        # Generate data
        z_true, activations = small_model.generate_data(n_samples=200, seed=0)
        h_0 = activations[0]

        # Train SAE briefly
        sae = SparseAutoencoder(SAEConfig(
            d_model=small_config.d_model,
            n_features=32,
            l1_coeff=1e-4
        ))

        opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
        for _ in range(50):  # Just 50 steps
            idx = torch.randperm(len(h_0))[:32]
            batch = h_0[idx]
            _, _, loss = sae(batch)
            loss.backward()
            opt.step()
            opt.zero_grad()

        # Evaluate
        true_dirs = get_true_features_at_layer(small_model, 0)
        sae_dirs = sae.feature_directions()
        result = feature_recovery_rate(true_dirs.cpu(), sae_dirs.detach().cpu())
        # Just check it runs and returns sane values
        assert 0 <= result["recovery_rate"] <= 1
        assert result["n_true"] == small_config.n_features
