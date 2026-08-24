import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse

LOG1P = r'E:/python/GRNC/data/MEScounts10426.h5ad'
EXP1  = r"D:\python\pmgcn\data\MEScounts10426_vgcn_admm_topK20_rawX_mask0.01_ms7.h5ad"

adata_log = sc.read(LOG1P)
adata_exp = sc.read(EXP1)

X_log = adata_log.X
X_exp = adata_exp.X

if sparse.issparse(X_log):
    X_log = X_log.toarray()
else:
    X_log = np.asarray(X_log)

if sparse.issparse(X_exp):
    X_exp = X_exp.toarray()
else:
    X_exp = np.asarray(X_exp)

print("LOG1P shape:", X_log.shape)
print("EXP1  shape:", X_exp.shape)

# -----------------------------
# zero positions in LOG1P
# -----------------------------
zero_mask = (X_log <= 1e-8)
n_zero = int(zero_mask.sum())
total = X_log.size
print("\n[Zero positions in LOG1P]")
print("n_zero =", n_zero)
print("zero fraction =", n_zero / total)

# values at those zero positions after imputation
imp_vals = X_exp[zero_mask]

print("\n[Imputed values at original zero positions]")
print("count =", imp_vals.size)
print("min =", float(np.min(imp_vals)))
print("max =", float(np.max(imp_vals)))
print("mean =", float(np.mean(imp_vals)))
print("median =", float(np.median(imp_vals)))
print("std =", float(np.std(imp_vals)))

qs = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
qvals = np.quantile(imp_vals, qs)
print("\nQuantiles of imputed values on original zeros:")
for q, v in zip(qs, qvals):
    print(f"{q:>5.2f} : {v:.6f}")

# -----------------------------
# how many zeros stayed zero / became positive
# -----------------------------
print("\n[How many original zeros were changed?]")
print("still zero (<=1e-8):", int((imp_vals <= 1e-8).sum()))
print("> 0:", int((imp_vals > 1e-8).sum()))
print("> 0.01:", int((imp_vals > 0.01).sum()))
print("> 0.1:", int((imp_vals > 0.1).sum()))
print("> 0.5:", int((imp_vals > 0.5).sum()))
print("> 1.0:", int((imp_vals > 1.0).sum()))

# fractions
print("\nFractions among original zeros:")
print("> 0:", float((imp_vals > 1e-8).mean()))
print("> 0.01:", float((imp_vals > 0.01).mean()))
print("> 0.1:", float((imp_vals > 0.1).mean()))
print("> 0.5:", float((imp_vals > 0.5).mean()))
print("> 1.0:", float((imp_vals > 1.0).mean()))

# -----------------------------
# verify nonzero positions unchanged
# -----------------------------
nonzero_mask = ~zero_mask
diff_nonzero = np.abs(X_exp[nonzero_mask] - X_log[nonzero_mask])

print("\n[Check nonzero positions]")
print("max abs diff on original nonzero positions =", float(diff_nonzero.max()))
print("mean abs diff on original nonzero positions =", float(diff_nonzero.mean()))

# -----------------------------
# save a summary table
# -----------------------------
summary = pd.DataFrame({
    "metric": [
        "n_zero_in_LOG1P",
        "zero_fraction_in_LOG1P",
        "imputed_min",
        "imputed_max",
        "imputed_mean",
        "imputed_median",
        "imputed_std",
        "frac_imputed_gt_0",
        "frac_imputed_gt_0.01",
        "frac_imputed_gt_0.1",
        "frac_imputed_gt_0.5",
        "frac_imputed_gt_1.0",
        "max_abs_diff_nonzero_positions",
        "mean_abs_diff_nonzero_positions",
    ],
    "value": [
        n_zero,
        n_zero / total,
        float(np.min(imp_vals)),
        float(np.max(imp_vals)),
        float(np.mean(imp_vals)),
        float(np.median(imp_vals)),
        float(np.std(imp_vals)),
        float((imp_vals > 1e-8).mean()),
        float((imp_vals > 0.01).mean()),
        float((imp_vals > 0.1).mean()),
        float((imp_vals > 0.5).mean()),
        float((imp_vals > 1.0).mean()),
        float(diff_nonzero.max()),
        float(diff_nonzero.mean()),
    ]
})

changed_vals = imp_vals[imp_vals > 1e-8]
print("changed count =", changed_vals.size)
print("min =", changed_vals.min())
print("max =", changed_vals.max())
print("mean =", changed_vals.mean())
print("median =", np.median(changed_vals))
print("q25, q50, q75, q90, q95 =", np.quantile(changed_vals, [0.25,0.5,0.75,0.9,0.95]))

out_csv = EXP1.replace(".h5ad", "_imputation_summary_vs_log1p.csv")
summary.to_csv(out_csv, index=False)
print("\nSaved summary to:", out_csv)