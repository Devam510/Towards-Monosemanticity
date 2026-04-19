import sys; sys.path.insert(0, '.')
import torch
from src.models.synthetic_transformer import SyntheticTransformer, SyntheticConfig, get_true_features_at_layer
from src.models.ccjfr import CCJFRConfig, CCJFR
from src.evaluation.ground_truth import feature_recovery_rate

L, D, F, seed, gamma, l1 = 3, 128, 256, 42, 0.1, 1e-3
SPARSITY_K = 5
n_repr = L + 1
torch.manual_seed(seed)

synth_config = SyntheticConfig(n_layers=L, d_model=D, n_features=200, sparsity_k=SPARSITY_K)
model = SyntheticTransformer(synth_config)

ccjfr_config = CCJFRConfig(
    n_layers=n_repr, d_model=D, n_features=F, k=SPARSITY_K, l1_coeff=l1,
    consist_coeff=gamma, embed_anchor_coeff=0.0, unembed_anchor_coeff=0.0,
    lr=1e-3, seed=seed, consist_anneal_steps=500
)
ccjfr = CCJFR(ccjfr_config)
optimizer = torch.optim.Adam(ccjfr.parameters(), lr=1e-3)

compute_fns = []
for l_idx in range(L):
    def make_fn(idx):
        Wi = model.layer_weights_in[idx]
        Wo = model.layer_weights_out[idx]
        return lambda h: h + torch.nn.functional.gelu(h @ Wi.T) @ Wo.T
    compute_fns.append(make_fn(l_idx))

ccjfr.train()
for step in range(300):
    z_true, activations = model.generate_data(n_samples=512)
    optimizer.zero_grad()
    loss, metrics = ccjfr(activations, compute_fns=compute_fns)
    loss.backward()
    optimizer.step()
    ccjfr.normalize_all_decoders()

ccjfr.eval()
recov = []
for l_idx in range(n_repr):
    sae_dirs = ccjfr.feature_directions_at(l_idx)
    true_dirs = get_true_features_at_layer(model, l_idx).data
    res = feature_recovery_rate(true_dirs, sae_dirs, threshold=0.85)
    recov.append(res["recovery_rate"])
    print("Layer", l_idx, "recovery:", round(res["recovery_rate"], 3), "mean_cos:", round(res["mean_max_cos"], 3))

print("Average recovery:", round(sum(recov)/len(recov), 3))
