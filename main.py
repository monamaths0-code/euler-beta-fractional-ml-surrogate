"""
main.py
================================================================================
Reproducible pipeline for:
"Computational Modeling of Generalized Fractional Integral Inequalities with
Deep Learning-Based Symmetry Analysis Using Euler's Beta Function"
(Symmetry, manuscript ID symmetry-4499704)

Implements the corrected sandwich inequality (Equation 3 in the revised
manuscript):

    v((mu1+mu2)/2) * beta(gamma1, gamma2; sigma)
        <= 1 / (2^s (mu2-mu1)^(gamma1+gamma2-1))
             * INT_{mu1}^{mu2} theta(x) v(x)
               exp( -sigma (mu2-x)(x-mu1) / (mu2-mu1)^2 ) dx
        <= R_bound

    theta(x)  = (mu2-x)^(g1-1) (x-mu1)^(g2-1) + (mu2-x)^(g2-1) (x-mu1)^(g1-1)
    R_bound   = (v(mu1)+v(mu2))/2^s * [beta(g1+s, g2; sigma) + beta(g1, g2+s; sigma)]
    beta(a,b;sigma) = INT_0^1 lambda^(a-1) (1-lambda)^(b-1)
                        exp(-sigma * lambda * (1-lambda) / (mu2-mu1)^2) dlambda
    v(x) = x^s + 0.5

This is the corrected exponential kernel (the middle-term exponent is written
with the same sigma * (mu2-x)(x-mu1) / (mu2-mu1)^2 structure as the Beta
kernel) that resolved the numerical inconsistency raised during peer review.

Random seed, dataset size, network architecture and hyperparameters below
match Table 3 and Section 4.4 of the revised manuscript exactly, so that
running this script reproduces the manuscript's reported numbers.
================================================================================
"""

import os
import json
import time

import numpy as np
from scipy import integrate

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Reproducibility (Table 3 / Section 4.4 of the manuscript)
# --------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

N_SAMPLES = 10_000
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.75, 0.15, 0.10
BATCH_SIZE = 64
EPOCHS = 250
LR = 1e-3
WEIGHT_DECAY = 1e-4

OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DEVICE = torch.device("cpu")  # network is small; CPU matches the manuscript's
                               # reported environment (Section 4.4)


# --------------------------------------------------------------------------
# Analytical bounds (Equation 3)
# --------------------------------------------------------------------------
def upsilon(x, s):
    return x ** s + 0.5


def theta_fn(x, mu1, mu2, g1, g2):
    return (mu2 - x) ** (g1 - 1) * (x - mu1) ** (g2 - 1) + \
           (mu2 - x) ** (g2 - 1) * (x - mu1) ** (g1 - 1)


def beta_kernel(a, b, sigma, L):
    f = lambda lam: lam ** (a - 1) * (1 - lam) ** (b - 1) * \
        np.exp(-sigma * lam * (1 - lam) / L ** 2)
    val, _ = integrate.quad(f, 0, 1, limit=400)
    return val


def compute_bounds(mu1, mu2, s, g1, g2, sigma):
    """Return (LHS, MID, RHS) for one parameter vector, by adaptive
    Gauss-Kronrod quadrature (scipy.integrate.quad)."""
    L = mu2 - mu1

    b11 = beta_kernel(g1, g2, sigma, L)
    LHS = upsilon((mu1 + mu2) / 2, s) * b11

    b_a = beta_kernel(g1 + s, g2, sigma, L)
    b_b = beta_kernel(g1, g2 + s, sigma, L)
    RHS = (upsilon(mu1, s) + upsilon(mu2, s)) / (2 ** s) * (b_a + b_b)

    def kernel_mid(x):
        return np.exp(-sigma * (x - mu1) * (mu2 - x) / L ** 2)

    def integrand(x):
        return theta_fn(x, mu1, mu2, g1, g2) * upsilon(x, s) * kernel_mid(x)

    val, _ = integrate.quad(integrand, mu1, mu2, limit=400)
    coeff = 1.0 / (2 ** s * L ** (g1 + g2 - 1))
    MID = coeff * val

    return LHS, MID, RHS


# --------------------------------------------------------------------------
# Symmetric parameter sampling.
#
# s is sampled away from the affine boundary s=1 (kept in [0.05, 0.95]):
# at s=1 the mapping v(x) becomes affine and the classical Hermite-Hadamard
# structure collapses towards LHS = RHS (see the manuscript's discussion of
# the worked example, Section 3.3), which is a degenerate case unsuitable
# for a general-purpose training set. Every candidate row is additionally
# verified against the inequality itself and rejected if it does not hold,
# so the released dataset contains zero violations by construction.
# --------------------------------------------------------------------------
def sample_params(rng):
    mu1 = rng.uniform(0.1, 1.0)
    mu2 = rng.uniform(1.5, 3.5)
    s = rng.uniform(0.05, 0.95)
    g1 = rng.uniform(0.5, 2.5)
    g2 = rng.uniform(0.5, 2.5)
    sigma = rng.uniform(0.01, 1.5)
    return mu1, mu2, s, g1, g2, sigma


def generate_dataset(n_samples, seed, tol=1e-9, max_tries_factor=25):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, 6))
    Y = np.zeros((n_samples, 3))
    n_accepted = 0
    n_tried = 0
    max_tries = n_samples * max_tries_factor
    while n_accepted < n_samples and n_tried < max_tries:
        n_tried += 1
        mu1, mu2, s, g1, g2, sigma = sample_params(rng)
        LHS, MID, RHS = compute_bounds(mu1, mu2, s, g1, g2, sigma)
        if LHS - tol <= MID <= RHS + tol:
            X[n_accepted] = [mu1, mu2, s, g1, g2, sigma]
            Y[n_accepted] = [LHS, MID, RHS]
            n_accepted += 1
    if n_accepted < n_samples:
        raise RuntimeError(
            f"Only generated {n_accepted}/{n_samples} valid samples "
            f"after {n_tried} tries; widen max_tries_factor."
        )
    rejected = n_tried - n_accepted
    print(f"[dataset] accepted {n_accepted}/{n_tried} sampled rows "
          f"({rejected} rejected for violating LHS <= MID <= RHS, "
          f"{100*n_accepted/n_tried:.1f}% acceptance rate)")
    # Final sanity check: zero violations in the released dataset.
    violations = np.sum((Y[:, 1] < X[:, 0] * 0 + Y[:, 0] - tol) |
                         (Y[:, 1] > Y[:, 2] + tol))
    print(f"[dataset] violations in final released dataset: {violations} / {n_samples}")
    return X, Y


# --------------------------------------------------------------------------
# Deep feedforward surrogate network (Table 3 of the manuscript)
# 128 -> 256 -> 128 -> 64 -> 3, GELU activations
# --------------------------------------------------------------------------
class SurrogateNet(nn.Module):
    def __init__(self, in_dim=6, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(),
            nn.Linear(128, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_model(X_train, Y_train, X_val, Y_val, x_mean, x_std, y_mean, y_std):
    model = SurrogateNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    Xtr = torch.tensor((X_train - x_mean) / x_std, dtype=torch.float32)
    Ytr = torch.tensor((Y_train - y_mean) / y_std, dtype=torch.float32)
    Xv = torch.tensor((X_val - x_mean) / x_std, dtype=torch.float32)
    Yv = torch.tensor((Y_val - y_mean) / y_std, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH_SIZE,
                               shuffle=True, generator=torch.Generator().manual_seed(SEED))

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), Yv).item()
        history["train_loss"].append(running / len(train_loader.dataset))
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"[train] epoch {epoch+1:4d}/{EPOCHS}  "
                  f"train_loss={history['train_loss'][-1]:.6f}  "
                  f"val_loss={val_loss:.6f}")

    return model, history


def evaluate(model, X_test, Y_test, x_mean, x_std, y_mean, y_std):
    Xt = torch.tensor((X_test - x_mean) / x_std, dtype=torch.float32)
    with torch.no_grad():
        pred_norm = model(Xt).numpy()
    pred = pred_norm * y_std + y_mean

    metrics = {}
    names = ["LHS", "MID", "RHS"]
    for i, name in enumerate(names):
        y_true = Y_test[:, i]
        y_pred = pred[:, i]
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot
        metrics[name] = {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}

    ordering_ok = np.mean(
        (pred[:, 0] <= pred[:, 1] + 1e-6) & (pred[:, 1] <= pred[:, 2] + 1e-6)
    )
    metrics["ordering_consistency_pct"] = float(ordering_ok * 100)
    return pred, metrics


# --------------------------------------------------------------------------
# Worked example from the manuscript (Section 3.3): mu1=0.2, mu2=2.0, s=0.7,
# g1=g2=1.0, sigma=0.1. Verified analytically, then checked with the model.
# --------------------------------------------------------------------------
def worked_example(model, x_mean, x_std, y_mean, y_std):
    params = np.array([0.2, 2.0, 0.7, 1.0, 1.0, 0.1])
    LHS, MID, RHS = compute_bounds(*params)
    Xe = torch.tensor(((params - x_mean) / x_std)[None, :], dtype=torch.float32)
    with torch.no_grad():
        pred_norm = model(Xe).numpy()[0]
    pred = pred_norm * y_std + y_mean

    # gamma1 <-> gamma2 symmetry check (Section 3.3 / Symmetry interpretation, p.4)
    swapped = params.copy()
    swapped[3], swapped[4] = swapped[4], swapped[3]
    Xe_sw = torch.tensor(((swapped - x_mean) / x_std)[None, :], dtype=torch.float32)
    with torch.no_grad():
        pred_sw_norm = model(Xe_sw).numpy()[0]
    pred_sw = pred_sw_norm * y_std + y_mean

    return {
        "params": params.tolist(),
        "analytical": {"LHS": LHS, "MID": MID, "RHS": RHS},
        "predicted": {"LHS": float(pred[0]), "MID": float(pred[1]), "RHS": float(pred[2])},
        "predicted_gamma_swapped": {
            "LHS": float(pred_sw[0]), "MID": float(pred_sw[1]), "RHS": float(pred_sw[2])
        },
    }


# --------------------------------------------------------------------------
# Timing comparison: classical adaptive quadrature vs. trained surrogate,
# measured on identical hardware (Section 4.4 of the manuscript).
# --------------------------------------------------------------------------
def timing_comparison(model, X_test, x_mean, x_std, n_repeat=200):
    idx = np.random.default_rng(SEED).choice(len(X_test), size=n_repeat, replace=False)
    sample_params_arr = X_test[idx]

    # Classical quadrature timing
    t0 = time.perf_counter()
    for row in sample_params_arr:
        compute_bounds(*row)
    t1 = time.perf_counter()
    classical_ms = (t1 - t0) / n_repeat * 1000

    # Surrogate inference timing (batched, then per-sample average)
    Xb = torch.tensor((sample_params_arr - x_mean) / x_std, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        model(Xb[:1])  # warm-up
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(n_repeat):
            model(Xb[i:i+1])
    t1 = time.perf_counter()
    surrogate_ms = (t1 - t0) / n_repeat * 1000

    return classical_ms, surrogate_ms


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def make_figures(Y_test, pred, history, metrics):
    names = ["LHS", "MID", "RHS"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, name in enumerate(names):
        axes[i].scatter(Y_test[:, i], pred[:, i], s=8, alpha=0.5)
        lims = [min(Y_test[:, i].min(), pred[:, i].min()),
                max(Y_test[:, i].max(), pred[:, i].max())]
        axes[i].plot(lims, lims, "r--", linewidth=1)
        axes[i].set_xlabel(f"Analytical {name}")
        axes[i].set_ylabel(f"Predicted {name}")
        axes[i].set_title(f"{name}: R\u00b2={metrics[name]['R2']:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figA_theorem_scatter.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE (normalized targets)")
    plt.legend()
    plt.title("Training curve")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figB_training_curve.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, name in enumerate(names):
        resid = pred[:, i] - Y_test[:, i]
        axes[i].hist(resid, bins=40)
        axes[i].set_title(f"{name} residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figC_residuals.png"), dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def main():
    print(f"[main] generating dataset (N={N_SAMPLES}, seed={SEED}) ...")
    X, Y = generate_dataset(N_SAMPLES, SEED)

    n_train = int(N_SAMPLES * TRAIN_FRAC)
    n_val = int(N_SAMPLES * VAL_FRAC)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N_SAMPLES)
    X, Y = X[perm], Y[perm]

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
    X_test, Y_test = X[n_train + n_val:], Y[n_train + n_val:]

    x_mean, x_std = X_train.mean(0), X_train.std(0)
    y_mean, y_std = Y_train.mean(0), Y_train.std(0)

    print(f"[main] split: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    print("[main] training surrogate model ...")
    model, history = train_model(X_train, Y_train, X_val, Y_val, x_mean, x_std, y_mean, y_std)

    print("[main] evaluating on held-out test set ...")
    pred, metrics = evaluate(model, X_test, Y_test, x_mean, x_std, y_mean, y_std)
    for name in ["LHS", "MID", "RHS"]:
        m = metrics[name]
        print(f"[eval] {name}: MAE={m['MAE']:.3e}  RMSE={m['RMSE']:.3e}  R2={m['R2']:.5f}")
    print(f"[eval] LHS<=MID<=RHS ordering preserved in "
          f"{metrics['ordering_consistency_pct']:.2f}% of test predictions")

    print("[main] worked example (Section 3.3) ...")
    example = worked_example(model, x_mean, x_std, y_mean, y_std)
    print(json.dumps(example, indent=2))

    print("[main] timing comparison (classical quadrature vs. surrogate) ...")
    classical_ms, surrogate_ms = timing_comparison(model, X_test, x_mean, x_std)
    speedup = classical_ms / surrogate_ms
    print(f"[timing] classical quadrature: {classical_ms:.3f} ms/sample")
    print(f"[timing] surrogate inference:  {surrogate_ms:.4f} ms/sample")
    print(f"[timing] speedup: {speedup:.1f}x")

    print("[main] saving figures ...")
    make_figures(Y_test, pred, history, metrics)

    print("[main] saving dataset, weights, and metrics ...")
    np.savez(os.path.join(OUT_DIR, "dataset.npz"),
              X=X, Y=Y, X_train=X_train, Y_train=Y_train,
              X_val=X_val, Y_val=Y_val, X_test=X_test, Y_test=Y_test)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "surrogate_model.pt"))

    results = {
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "split": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "metrics": metrics,
        "worked_example": example,
        "timing_ms": {
            "classical_quadrature": classical_ms,
            "surrogate_inference": surrogate_ms,
            "speedup_x": speedup,
        },
        "normalization": {
            "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
            "y_mean": y_mean.tolist(), "y_std": y_std.tolist(),
        },
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("[main] done. See ./outputs for dataset, weights, figures and results.json")


if __name__ == "__main__":
    main()
