# euler-beta-fractional-ml-surrogate

Reproducible pipeline for the paper **"Computational Modeling of Generalized
Fractional Integral Inequalities with Deep Learning-Based Symmetry Analysis
Using Euler's Beta Function"** (submitted to *Symmetry*, manuscript ID
symmetry-4499704).

This repository is specific to that manuscript. It is not the same project as
[`euler-beta-ml-fractional-inequalities`](https://github.com/monamaths0-code/euler-beta-ml-fractional-inequalities),
which covers four separate theorems from Almutairi (2023) — this repository
covers the single generalized sandwich inequality (with the γ₁, γ₂, σ kernel
and the γ₁ ↔ γ₂ symmetry) that is the subject of the Symmetry submission.

## What this reproduces

For the inequality

```
v((μ1+μ2)/2) · β(γ1, γ2; σ)
    ≤ 1/(2^s (μ2-μ1)^(γ1+γ2-1)) ∫_{μ1}^{μ2} ϑ(x) v(x) exp(-σ(μ2-x)(x-μ1)/(μ2-μ1)²) dx
    ≤ R_bound
```

(Equation 3 of the manuscript, with the corrected exponential kernel — see
"Note on the corrected kernel" below), the pipeline:

1. Generates a synthetic dataset of 10,000 parameter vectors
   `(μ1, μ2, s, γ1, γ2, σ)`, computing the exact LHS, MID, and RHS bounds for
   each by adaptive Gauss–Kronrod quadrature (`scipy.integrate.quad`).
2. Rejects and re-samples any candidate row that does not satisfy
   `LHS ≤ MID ≤ RHS`, so the released dataset contains **zero violations** by
   construction.
3. Trains a deep feedforward surrogate network (128 → 256 → 128 → 64, GELU
   activations, AdamW optimizer, cosine annealing schedule) to predict the
   three bounds directly from the six input parameters.
4. Evaluates the surrogate on a held-out test partition (MAE, RMSE, R²), and
   checks how often the predicted ordering `LHS̑ ≤ MID̑ ≤ RHS̑` is preserved.
5. Reproduces the manuscript's worked example (Section 3.3) and its
   γ₁ ↔ γ₂ symmetry check.
6. Times classical quadrature against surrogate inference on the same
   machine, for the reported speed-up.


## Repository structure

```
.
├── main.py              # Full pipeline: dataset generation, training, evaluation, figures
├── requirements.txt      # Python dependencies
├── outputs/              # Created by running main.py
│   ├── dataset.npz            # Generated dataset (train/val/test splits)
│   ├── surrogate_model.pt     # Trained network weights
│   ├── results.json           # Metrics, worked example, timing comparison
│   └── figures/
│       ├── figA_theorem_scatter.png
│       ├── figB_training_curve.png
│       └── figC_residuals.png
└── README.md
```

## Requirements

```
pip install -r requirements.txt
```

Tested with Python 3.10+. The network is small enough to train on CPU only
(no GPU required); a full run (dataset generation + 250-epoch training +
evaluation) takes a few minutes on a standard laptop CPU.

## Usage

```
python main.py
```

This will generate the dataset, train the surrogate model with the fixed
random seed used in the manuscript (`seed = 42`), evaluate it, print the
worked example and timing comparison to the console, and save everything
under `./outputs`.

## Reproducibility

All the numbers below come from an actual run of `main.py` in this
repository  and correspond to the numbers reported in
the  manuscript (Table 2, Table 3, Table 4, and Section 4 of the
paper):

| Bound | MAE | RMSE | R² |
|---|---|---|---|
| LHS | 1.23×10⁻³ | 2.06×10⁻³ | 0.99999 |
| MID | 1.43×10⁻³ | 2.37×10⁻³ | 1.00000 |
| RHS | 3.17×10⁻³ | 5.76×10⁻³ | 0.99999 |

- Ordering (`LHS̑ ≤ MID̑ ≤ RHS̑`) preserved on **99.90%** of the held-out test set.
- Classical adaptive quadrature: **≈2.30 ms/sample**; surrogate inference:
  **≈0.12 ms/sample** (≈20× speed-up), both measured on the same CPU.

Exact timing numbers are hardware- and implementation-dependent and will
vary somewhat between machines; re-running `main.py` on a different machine
will reproduce the same dataset, training curve, and accuracy metrics (fixed
seed), but the millisecond-level timing numbers are only representative of
the machine they were measured on.

## Citation

If you use this code, please cite the manuscript:

```
@article{siddiq2026symmetry,
  title   = {Computational Modeling of Generalized Fractional Integral
             Inequalities with Deep Learning-Based Symmetry Analysis
             Using Euler's Beta Function},
  journal = {Symmetry},
  year    = {2026},
  note    = {Manuscript ID symmetry-4499704}
}
```

## Contact

Mamoona Siddiq — Department of Mathematics and Statistics, University of
Lahore, Sargodha Campus, Pakistan.

## License

MIT License (see LICENSE).
