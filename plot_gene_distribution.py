import scanpy as sc
import numpy as np
from utils import to_dense
import matplotlib.pyplot as plt
RAW = r"E:\python\GRNC\data\MEScounts_raw.h5ad"
EXP1 = r"D:\python\pmgcn\data\MEScounts10426_vgcn_admm_topK30_rawX_mask0.01_ms7.h5ad"

adata = sc.read(RAW)
#X_true = to_dense(adata.X).astype(np.float64)
print(adata)


gene = np.random.choice(adata.var_names)
gene = "Abhd12"  # 举例
print("Random gene:", gene)
x = adata[:, gene].X
x = x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()

plt.figure()
plt.hist(x, bins=50)
plt.title(f"{gene}: overall (incl. zeros)")
plt.xlabel("Expression")
plt.ylabel("Cells")
plt.show()

x_pos = x[x > 0]
plt.figure()
plt.hist(x_pos, bins=50)
plt.title(f"{gene}: positive-only (x>0)")
plt.xlabel("Expression (positive)")
plt.ylabel("Cells")
plt.show()
