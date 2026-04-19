import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def setup_academic_style():
    """Configures matplotlib for NeurIPS/ICML style plots."""
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (5.5, 4),
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })
    sns.set_palette("colorblind")

def plot_gamma_sweep(df: pd.DataFrame, output_dir: Path):
    setup_academic_style()
    df_gamma = df[df["sweep"] == "gamma"]
    if df_gamma.empty: return
    
    fig, ax1 = plt.subplots()
    
    df_mean = df_gamma.groupby("gamma").mean(numeric_only=True).reset_index()
    df_std = df_gamma.groupby("gamma").std(numeric_only=True).reset_index()
    
    ax1.set_xlabel(r"Consistency Coefficient ($\gamma$)")
    ax1.set_ylabel("Feature Recovery (%)", color="C0")
    ax1.plot(df_mean["gamma"], df_mean["recovery"] * 100, color="C0", marker="o", label="Recovery")
    
    # Fill between if std is not NaN
    if not df_std["recovery"].isna().any():
        ax1.fill_between(df_mean["gamma"], 
                         (df_mean["recovery"] - df_std["recovery"])*100, 
                         (df_mean["recovery"] + df_std["recovery"])*100, color="C0", alpha=0.2)
    ax1.tick_params(axis='y', labelcolor="C0")
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Reconstruction MSE", color="C1")
    ax2.plot(df_mean["gamma"], df_mean["mse"], color="C1", marker="x", linestyle="--", label="MSE")
    ax2.tick_params(axis='y', labelcolor="C1")
    
    plt.title(r"Impact of $\gamma$ on CCJFR Performance")
    fig.tight_layout()
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "gamma_vs_convergence.pdf", bbox_inches="tight")
    plt.close()

def plot_layer_count_sweep(df: pd.DataFrame, output_dir: Path):
    setup_academic_style()
    df_layers = df[df["sweep"] == "layer_count"]
    if df_layers.empty: return
    
    fig, ax = plt.subplots()
    df_mean = df_layers.groupby("L").mean(numeric_only=True).reset_index()
    
    ax.plot(df_mean["L"], df_mean["recovery"] * 100, marker="s", color="C2")
    ax.set_xlabel("Number of Layers (L)")
    ax.set_ylabel("Feature Recovery (%)")
    ax.set_title("Recovery Scaling with CCJFR Depth")
    
    ax.set_xticks(df_mean["L"].unique())
    
    fig.tight_layout()
    plt.savefig(output_dir / "layer_count_vs_recovery.pdf", bbox_inches="tight")
    plt.close()


def plot_lambda_sweep(df: pd.DataFrame, output_dir: Path):
    """Plot feature recovery vs sparsity penalty (lambda)."""
    setup_academic_style()
    df_lam = df[df["sweep"] == "lambda"]
    if df_lam.empty: return

    fig, ax = plt.subplots()
    ax.plot(df_lam["lambda"], df_lam["recovery"] * 100, marker="D", color="C3")
    ax.set_xlabel(r"Sparsity Penalty ($\lambda$)")
    ax.set_ylabel("Feature Recovery (%)")
    ax.set_title(r"Impact of $\lambda$ on Feature Recovery")
    ax.set_xscale("log")

    fig.tight_layout()
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "lambda_vs_recovery.pdf", bbox_inches="tight")
    plt.close()


def compute_failure_modes(df: pd.DataFrame) -> dict:
    """
    Identify and document failure modes from ablation results.

    Returns a dict summarising:
    - Optimal gamma range (window of gamma where recovery is maximal)
    - Gamma values that cause divergence (recovery < 0.5)
    - Lambda values where sparsity kills recovery
    - Whether more layers -> strictly better recovery
    """
    report = {}

    # --- Gamma analysis ---
    df_gamma = df[df["sweep"] == "gamma"].groupby("gamma").mean(numeric_only=True)
    if not df_gamma.empty:
        best_gamma = df_gamma["recovery"].idxmax()
        fail_gammas = df_gamma[df_gamma["recovery"] < 0.5].index.tolist()
        report["optimal_gamma"] = best_gamma
        report["best_gamma_recovery"] = float(df_gamma.loc[best_gamma, "recovery"])
        report["diverging_gammas"] = fail_gammas
        report["gamma_failure_mode"] = (
            "High gamma (>= 0.5) over-constrains reconstruction, collapsing recovery."
            if any(g >= 0.5 for g in fail_gammas) else
            "No gamma divergence detected in tested range."
        )

    # --- Layer count analysis ---
    df_layers = df[df["sweep"] == "layer_count"].groupby("L").mean(numeric_only=True)
    if not df_layers.empty:
        recovery_vals = df_layers["recovery"].values
        monotone_increasing = all(
            recovery_vals[i] <= recovery_vals[i+1] for i in range(len(recovery_vals)-1)
        )
        report["more_layers_better"] = monotone_increasing
        report["layer_recovery_trend"] = df_layers["recovery"].to_dict()

    # --- Lambda analysis ---
    df_lam = df[df["sweep"] == "lambda"].groupby("lambda").mean(numeric_only=True)
    if not df_lam.empty:
        best_lambda = df_lam["recovery"].idxmax()
        report["optimal_lambda"] = best_lambda
        report["lambda_failure_mode"] = (
            "Very high lambda kills recovery by over-penalising features."
            if df_lam["recovery"].iloc[-1] < df_lam["recovery"].max() * 0.8 else
            "Lambda robust in tested range."
        )

    return report
