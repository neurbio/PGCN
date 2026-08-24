# PGCN: Precision-Matrix-Guided Graph Convolution for Single-Cell RNA-seq Imputation

This repository contains the source code associated with the manuscript:

**Precision-Matrix-Guided Graph Convolution for Single-Cell RNA-Seq Imputation**

PGCN constructs a cell-cell graph using sparse precision-matrix estimation and uses graph convolution together with a Two-Part Generalized Gamma (TPGG) decoder for scRNA-seq imputation.

## Overview

The main PGCN pipeline consists of the following steps:

1. Preprocess the scRNA-seq expression matrix using library-size normalization and `log1p` transformation.
2. Compute pairwise Pearson correlations between cells.
3. Apply correlation shrinkage for numerical stability.
4. Estimate a sparse precision matrix using ADMM.
5. Convert precision-matrix entries to partial-correlation weights.
6. Construct a sparse cell-cell graph using mutual Top-K neighbors.
7. Use the graph convolutional encoder to learn cell representations.
8. Decode the learned representations using the TPGG-based decoder.
9. Selectively impute candidate dropout entries while preserving observed nonzero expression values.

## Repository Structure

```text
PGCN/
├── main.py                # Main entry point for graph construction, training, and imputation
├── ADMM_surrogate.py      # Sparse precision-matrix estimation using ADMM
├── network.py             # Graph convolutional encoder and TPGG decoder
├── layers.py              # Graph convolution and sparse adjacency utilities
├── loss.py                # TPGG and training loss functions
├── train.py               # Model training and inference
├── utils.py               # General utilities and evaluation helpers
├── data.py                # Dataset-specific data preparation examples
├── eval.py                # Imputation diagnostic analysis
├── test.py                # Additional downstream diagnostic analysis
└── preprocess.py          # Exploratory expression-distribution analysis
```

## Requirements

The implementation is written in Python and uses TensorFlow, Scanpy, and SciPy.

The main dependencies are:

```text
Python
tensorflow
numpy
scipy
pandas
anndata
scanpy
scikit-learn
matplotlib
openpyxl
```

We recommend creating a separate virtual environment before installation.

```bash
python -m venv pgcn_env
```

On Linux/macOS:

```bash
source pgcn_env/bin/activate
```

On Windows:

```bash
pgcn_env\Scripts\activate
```

Install the required Python packages:

```bash
pip install tensorflow numpy scipy pandas anndata scanpy scikit-learn matplotlib openpyxl
```

For exact reproducibility, the package versions used for the reported experiments should be recorded in `requirements.txt`.

## Data

The real scRNA-seq datasets used in the study are publicly available from the **National Center for Biotechnology Information (NCBI) Gene Expression Omnibus (GEO)**.

| Dataset | GEO accession | Repository |
|---|---|---|
| Klein | GSE65525 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525 |
| Chu | GSE75748 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75748 |
| Romanov | GSE74672 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE74672 |

The simulated dataset described in the manuscript was generated using the **Splatter** R package.

Large expression datasets are not stored directly in this repository. Users should download the original datasets from the repositories above and convert them to the AnnData (`.h5ad`) format before running PGCN.

## Input Format

The main program accepts an AnnData (`.h5ad`) file:

```text
cells × genes
```

The expression matrix is expected to contain non-negative scRNA-seq expression values.

The recommended preprocessing is:

```python
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
```

`main.py` also contains preprocessing checks and can apply normalization and `log1p` transformation when necessary.

For maximum reproducibility, we recommend explicitly preprocessing the input dataset before running PGCN.

## Running PGCN

The main entry point is:

```bash
python main.py
```

A typical strict-mask evaluation run is:

```bash
python main.py \
    --input ./data/input.h5ad \
    --out_dir ./results \
    --topK 20 \
    --shrink_alpha 0.001 \
    --gl_lam 0.001 \
    --gl_rho 0.01 \
    --admm_max_iter 500 \
    --epochs 200 \
    --lr 0.001 \
    --hidden1 256 \
    --hidden2 128 \
    --latent_dim 64 \
    --dropout 0.1 \
    --x_drop_prob 0.2 \
    --apply_mask \
    --mask_prob 0.01 \
    --mask_seed 0 \
    --seed 0
```

On Windows PowerShell or Command Prompt, the command can also be entered on a single line.

## Main Parameters

| Parameter | Default | Description |
|---|---:|---|
| `--input` | — | Input `.h5ad` file |
| `--out_dir` | `./data` | Output directory |
| `--topK` | 20 | Number of neighbors used for mutual Top-K graph construction |
| `--shrink_alpha` | 0.001 | Correlation shrinkage parameter |
| `--gl_lam` | 0.001 | ADMM sparsity parameter |
| `--gl_rho` | 0.01 | ADMM penalty parameter |
| `--admm_max_iter` | 500 | Maximum number of ADMM iterations |
| `--epochs` | 200 | Number of training epochs |
| `--lr` | 0.001 | Adam learning rate |
| `--hidden1` | 256 | First hidden dimension |
| `--hidden2` | 128 | Decoder hidden dimension |
| `--latent_dim` | 64 | Latent cell representation dimension |
| `--dropout` | 0.1 | Neural-network dropout rate |
| `--x_drop_prob` | 0.2 | Positive-expression corruption probability during denoising training |
| `--seed` | 0 | Random seed |
| `--apply_mask` | disabled | Enable strict holdout evaluation |
| `--mask_prob` | 0.01 | Positive-entry holdout probability |
| `--mask_seed` | 0 | Random seed for strict masking |
| `--use_svd_feat` | disabled | Use SVD-reduced features instead of the full expression matrix |
| `--k_use` | 200 | SVD rank when `--use_svd_feat` is enabled |

## Precision-Matrix Graph Construction

Given the cell-feature matrix, PGCN first computes a cell-cell Pearson correlation matrix.

A shrinkage estimator is then applied:

```text
S_alpha = (1 - alpha) S + alpha I
```

A sparse precision matrix is estimated using ADMM. The precision matrix is converted to absolute partial-correlation weights:

```text
rho_ij = -Theta_ij / sqrt(Theta_ii Theta_jj)
```

The final adjacency matrix is constructed using mutual Top-K neighbors.

The resulting graph is stored in `cache_graph/` so that it can be reused for subsequent runs with identical graph-construction settings.

## TPGG Decoder

The decoder predicts:

```text
pi0 : zero-component probability
a   : generalized-gamma scale parameter
d   : generalized-gamma shape parameter
p   : generalized-gamma family parameter
```

For the positive component, the generalized-gamma conditional mean is

```text
mu_GG = a × exp[lgamma(d + 1/p) - lgamma(d)].
```

The zero-probability branch and positive-expression branch are jointly used to determine the imputed expression values.

## Strict Holdout Evaluation

For controlled evaluation, the program can mask a subset of originally observed positive entries and a small subset of zero entries.

Enable this mode using:

```bash
--apply_mask --mask_prob 0.01
```

The holdout entries are excluded from the standard reconstruction mask and are used to evaluate recovery of positive expression values and preservation of zero entries.

Reported diagnostics include RMSE, MAE, zero-entry leakage, and the behavior of the zero-probability gate.

## Output

The main program writes an imputed AnnData file to the specified output directory.

The output contains:

```text
adata.X
    Final expression matrix after selective imputation.

adata.obsm["X_vgcn_mu"]
    Learned low-dimensional cell representation.

adata.uns
    Graph-construction, model, masking, and random-seed parameters.
```

The output filename records the input dataset, graph setting, feature mode, and holdout configuration.

## Reproducibility

Random seeds can be specified through:

```bash
--seed
--mask_seed
```

For comparisons across experimental settings, the same preprocessing procedure, random seeds, graph parameters, and training settings should be used.

Because the ADMM precision calculation and model training can be computationally intensive, runtime depends on the number of cells, number of genes, available memory, and TensorFlow hardware configuration.

## Source Code Availability

The source code supporting the study is publicly available at:

https://github.com/neurbio/PGCN

## Citation

If you use this code, please cite:

**Precision-Matrix-Guided Graph Convolution for Single-Cell RNA-Seq Imputation**

A complete bibliographic citation will be added after publication.

## Contact

Questions about the implementation can be submitted through the GitHub Issues page of this repository.