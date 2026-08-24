'''import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scanpy as sc
import numpy as np

# input / output
csv_path = r"E:\Input\GSE75748_sc_cell_type_ec.csv\GSE75748_sc_cell_type_ec.csv"
raw_h5ad_path = r"E:\Input\GSE75748_sc_cell_type_ec.h5ad"
log1p_h5ad_path = r"E:\Input\GSE75748_sc_cell_type_ec_lg1pdata.h5ad"

# read matrix: rows = genes, columns = cells
df = pd.read_csv(csv_path, index_col=0)

# transpose to Scanpy format: cells x genes
X = df.T

# build obs (cell metadata)
obs = pd.DataFrame(index=X.index)
obs["cell_id"] = obs.index
obs["cell_type"] = obs.index.str.extract(r"^([^_]+)", expand=False)   # H1, H9, NPC, ...
obs["batch"] = obs.index.str.extract(r"^[^_]+_([^.]+)", expand=False) # Exp1, Batch1, Batch2...

# broader grouped labels
obs["cell_type_broad"] = obs["cell_type"].replace({
    "H1": "ESC",
    "H9": "ESC"
})

# build var (gene metadata)
var = pd.DataFrame(index=X.columns)
var["gene_symbol"] = var.index

# convert to sparse matrix to save space
X_sparse = sp.csr_matrix(X.values)

# create AnnData
adata = ad.AnnData(X=X_sparse, obs=obs, var=var)

# make names unique just in case
adata.obs_names_make_unique()
adata.var_names_make_unique()

# save raw counts in layer
adata.layers["counts"] = adata.X.copy()

# save raw h5ad first
adata.write_h5ad(raw_h5ad_path, compression="gzip")

print("Raw AnnData:")
print(adata)
print("\nCell type counts:")
print(adata.obs["cell_type"].value_counts().sort_index())
print(f"\nSaved raw h5ad to: {raw_h5ad_path}")

# ---------------- preprocessing ----------------
# basic preprocessing for Scanpy:
# 1) library-size normalize each cell
# 2) log1p transform

adata_pp = adata.copy()

# normalize total counts per cell to 1e4
sc.pp.normalize_total(adata_pp, target_sum=1e4)

# log(1 + x)
sc.pp.log1p(adata_pp)

# mark preprocessing info
adata_pp.uns["log1p"] = {"base": None}
adata_pp.uns["normalized"] = {"target_sum": 1e4}

# optional: keep raw normalized input before HVG/scaling
adata_pp.raw = adata_pp

# save processed h5ad
adata_pp.write_h5ad(log1p_h5ad_path, compression="gzip")

print("\nPreprocessed AnnData:")
print(adata_pp)
print("min/max after log1p:", float(adata_pp.X.min()), float(adata_pp.X.max()))
print(f"\nSaved log1p h5ad to: {log1p_h5ad_path}")'''
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scanpy as sc
import numpy as np

xlsx_path = r"E:\Input\GSE74672_expressed_mols_with_classes.xlsx\hypoth_moldata_classification08-Mar-2017.xlsx"
h5ad_path = r"E:\Input\GSE74672_raw.h5ad"
log1p_h5ad_path = r"E:\Input\GSE74672_lg1p.h5ad"

# read whole sheet
df = pd.read_excel(xlsx_path, sheet_name=0, header=None)

# -------------------------------------------------
# file structure:
# col 0 = row names
# row 0 = cell IDs
# rows 1~... = metadata
# gene expression starts from row 12 (Excel row 13)
# -------------------------------------------------
gene_start_row = 12

# ---------- cell ids ----------
cell_ids = df.iloc[0, 1:].astype(str).values

# ---------- obs metadata ----------
obs = pd.DataFrame(index=cell_ids)
obs["cell_id"] = cell_ids

meta_map = {
    1: "level1_class",
    2: "level2_class",
    3: "level2_cluster_number",
    4: "age_postnatal",
    5: "sex",
    6: "cell_diameter",
    7: "acute_stress",
    8: "total_molecules",
}

for row_idx, col_name in meta_map.items():
    if row_idx < df.shape[0]:
        obs[col_name] = df.iloc[row_idx, 1:].values

# optional broader label
if "level1_class" in obs.columns:
    obs["cell_type"] = obs["level1_class"]
else:
    obs["cell_type"] = "unknown"

# ---------- fix obs dtypes ----------
# object columns with mixed numbers/strings/nan will fail when writing h5ad
# so convert each column explicitly

string_cols = ["cell_id", "level1_class", "level2_class", "age_postnatal", "sex", "acute_stress", "cell_type"]
numeric_cols = ["level2_cluster_number", "cell_diameter", "total_molecules"]

for col in string_cols:
    if col in obs.columns:
        obs[col] = obs[col].astype(str)
        obs[col] = obs[col].replace({"nan": "NA", "None": "NA"})

for col in numeric_cols:
    if col in obs.columns:
        obs[col] = pd.to_numeric(obs[col], errors="coerce")

# ---------- gene expression ----------
gene_df = df.iloc[gene_start_row:, :].copy()

gene_names = gene_df.iloc[:, 0].astype(str).values
expr = gene_df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0)

# transpose to cells x genes
X = expr.T
X.index = cell_ids
X.columns = gene_names

# ---------- var metadata ----------
var = pd.DataFrame(index=X.columns)
var["gene_symbol"] = var.index.astype(str)

# convert to sparse float32 matrix
X_sparse = sp.csr_matrix(X.values.astype(np.float32))

# create AnnData
adata = ad.AnnData(X=X_sparse, obs=obs, var=var, dtype=np.float32)

# make names unique
adata.obs_names_make_unique()
adata.var_names_make_unique()

# keep raw counts
adata.layers["counts"] = adata.X.copy()

# save raw
adata.write_h5ad(h5ad_path, compression="gzip")

print("Raw AnnData:")
print(adata)
print("\nobs dtypes:")
print(adata.obs.dtypes)
print("\nobs columns:")
print(list(adata.obs.columns))
print("\nSaved raw h5ad to:", h5ad_path)

# ---------- preprocessing: normalize + log1p ----------
adata_pp = adata.copy()

# keep raw snapshot
adata_pp.raw = adata.copy()

# normalize each cell to target_sum
sc.pp.normalize_total(adata_pp, target_sum=1e4)

# log1p
sc.pp.log1p(adata_pp)

# record preprocessing info
adata_pp.uns["log1p"] = {"base": None}
adata_pp.uns["normalized"] = {"target_sum": 1e4}

# save processed
adata_pp.write_h5ad(log1p_h5ad_path, compression="gzip")

print("\nPreprocessed AnnData:")
print(adata_pp)

if sp.issparse(adata_pp.X):
    Xmin = float(adata_pp.X.min())
    Xmax = float(adata_pp.X.max())
else:
    Xmin = float(np.min(adata_pp.X))
    Xmax = float(np.max(adata_pp.X))

print("min/max after log1p:", Xmin, Xmax)
print("\nSaved log1p h5ad to:", log1p_h5ad_path)