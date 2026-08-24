import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

LOG1P = r'E:/python/GRNC/data/MEScounts10426.h5ad'
EXP1  = r"D:\python\pmgcn\data\MEScounts10426_vgcn_admm_topK20_rawX_mask0.01_ms7.h5ad"
EXP2  = r"D:\python\pmgcn\data\MEScounts10426_vgcn_admm_topK30_rawX_mask0.01_ms7.h5ad"

LOG1P2 = r"E:\Input\GSE75748_19097"
EXP22  = r"D:\python\pmgcn\data\GSE75748_19097_gcn_admm_topK30_rawX_mask0.01_ms7.h5ad"

adata_log = sc.read(LOG1P2)
adata_exp = sc.read(EXP22)

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

# labels

'''y = np.load(r"E:/python/GRNC/data/mse_labels.npy")
lf = LabelEncoder().fit(y)
y_true = lf.transform(y)'''
label_col = "cell_type_broad"   # or "cell_type_broad"

if label_col not in adata_log.obs.columns:
    raise ValueError(f"'{label_col}' not found in adata.obs.columns: {list(adata.obs.columns)}")

y = adata_log.obs[label_col].astype(str).values
lf = LabelEncoder()
y_true = lf.fit_transform(y)

# focus on true_1 and true_2
mask1 = (y_true == 1)
mask2 = (y_true == 2)

print("n(true_1) =", int(mask1.sum()))
print("n(true_2) =", int(mask2.sum()))

# gene names
if adata_log.var_names is not None:
    gene_names = np.asarray(adata_log.var_names)
else:
    gene_names = np.array([f"gene_{i}" for i in range(X_log.shape[1])])

# class-wise mean
mean1_log = X_log[mask1].mean(axis=0)
mean2_log = X_log[mask2].mean(axis=0)

mean1_exp = X_exp[mask1].mean(axis=0)
mean2_exp = X_exp[mask2].mean(axis=0)

# class-wise zero fraction
zero1_log = (X_log[mask1] <= 1e-8).mean(axis=0)
zero2_log = (X_log[mask2] <= 1e-8).mean(axis=0)

zero1_exp = (X_exp[mask1] <= 1e-8).mean(axis=0)
zero2_exp = (X_exp[mask2] <= 1e-8).mean(axis=0)

# difference shrinkage
diff_log = mean1_log - mean2_log
diff_exp = mean1_exp - mean2_exp

abs_diff_log = np.abs(diff_log)
abs_diff_exp = np.abs(diff_exp)
shrink = abs_diff_log - abs_diff_exp   # positive => pulled closer after imputation

# how much expression was added on original zeros
orig_zero_mask = (X_log <= 1e-8)
imp_added = np.where(orig_zero_mask, X_exp, 0.0)

added1_mean = imp_added[mask1].mean(axis=0)
added2_mean = imp_added[mask2].mean(axis=0)
added_gap = np.abs(added1_mean - added2_mean)

# collect results
df = pd.DataFrame({
    "gene": gene_names,
    "mean1_log": mean1_log,
    "mean2_log": mean2_log,
    "mean1_exp": mean1_exp,
    "mean2_exp": mean2_exp,
    "diff_log": diff_log,
    "diff_exp": diff_exp,
    "abs_diff_log": abs_diff_log,
    "abs_diff_exp": abs_diff_exp,
    "shrink": shrink,
    "zero1_log": zero1_log,
    "zero2_log": zero2_log,
    "zero1_exp": zero1_exp,
    "zero2_exp": zero2_exp,
    "added1_mean": added1_mean,
    "added2_mean": added2_mean,
    "added_gap": added_gap,
})

# optional filtering:
# keep genes that had at least some separation originally
df_use = df[df["abs_diff_log"] > 0.2].copy()

# most pulled closer
df_closer = df_use.sort_values(["shrink", "abs_diff_log"], ascending=[False, False]).copy()

print("\nTop genes pulled closer between true_1 and true_2 after imputation:")
print(df_closer[[
    "gene",
    "mean1_log", "mean2_log",
    "mean1_exp", "mean2_exp",
    "abs_diff_log", "abs_diff_exp", "shrink",
    "added1_mean", "added2_mean"
]].head(30).to_string(index=False))

# save full results
#out_csv = EXP1.replace(".h5ad", "_true1_vs_true2_gene_shrink.csv")
#df_closer.to_csv(out_csv, index=False)
#print("\nSaved gene shrink table to:", out_csv)

print(df_closer[[
    "gene",
    "zero1_log", "zero2_log",
    "zero1_exp", "zero2_exp",
    "added1_mean", "added2_mean",
    "abs_diff_log", "abs_diff_exp", "shrink"
]].head(30).to_string(index=False))