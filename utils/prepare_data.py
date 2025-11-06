"""
prepare_data.py
-----------------------------------
Utility functions for DETECT:
- Synthetic Gaussian graph generation
- Covariance computation and conditioning
- Data normalization and validation

Author: DETECT Team (Dianjun Lin et al.)
"""

import numpy as np
import pandas as pd
import networkx as nx
from time import time
from scipy.stats import chi2_contingency, pearsonr
from sklearn import covariance
import torch


##############################
# 1️⃣ Data Simulation
##############################

def generate_random_graph(num_nodes, sparsity, seed=None):
    """Generate a random undirected Erdos–Rényi graph."""
    min_s, max_s = sparsity
    s = np.random.uniform(min_s, max_s)
    G = nx.erdos_renyi_graph(num_nodes, s, seed=seed)
    return nx.adjacency_matrix(G).todense()


def simulate_gaussian_samples(
    num_nodes,
    edge_connections,
    num_samples,
    seed=None,
    eig_offset=0.1,
    w_min=0.5,
    w_max=1.0,
):
    """Simulate multivariate Gaussian samples given graph structure."""
    if seed:
        np.random.seed(seed)

    # Precision matrix construction
    U = np.random.uniform(w_min, w_max, (num_nodes, num_nodes))
    theta = np.multiply(edge_connections, U)
    theta = (theta + theta.T) / 2 + np.eye(num_nodes)

    # Ensure positive-definiteness
    min_eval = np.min(np.linalg.eigvals(theta).real)
    precision = theta + np.eye(num_nodes) * (eig_offset - min_eval)
    cov = np.linalg.inv(precision)

    data = np.random.multivariate_normal(np.zeros(num_nodes), cov, size=num_samples)
    return data, precision


def get_data(
    num_nodes,
    sparsity,
    num_samples,
    batch_size=1,
    eig_offset=0.1,
    w_min=0.5,
    w_max=1.0,
):
    """Generate Gaussian data batches and true precision matrices."""
    Xb, true_theta = [], []
    for _ in range(batch_size):
        edges = generate_random_graph(num_nodes, sparsity)
        X, theta = simulate_gaussian_samples(
            num_nodes, edges, num_samples, eig_offset=eig_offset, w_min=w_min, w_max=w_max
        )
        Xb.append(X)
        true_theta.append(theta)
    return np.array(Xb), np.array(true_theta)


def add_noise_dropout(Xb, dropout=0.25):
    """Randomly replace a fraction of values with NaN to simulate missing data."""
    Xb_noisy = []
    for X in Xb:
        X_flat = X.reshape(-1)
        mask = np.random.choice(X_flat.size, int(X_flat.size * dropout), replace=False)
        X_flat[mask] = np.nan
        Xb_noisy.append(X_flat.reshape(X.shape))
    return np.array(Xb_noisy)


##############################
# 2️⃣ Data Processing
##############################

def convert_to_torch(data, req_grad=False, use_cuda=False):
    """Convert NumPy array to torch.Tensor."""
    if not torch.is_tensor(data):
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        data = torch.as_tensor(data, dtype=torch.float32).type(dtype)
    data.requires_grad = req_grad
    return data


def eig_condition_number(A):
    """Compute eigenvalues and condition number of matrix."""
    eigvals = np.linalg.eigvals(A).real
    cond = np.max(np.abs(eigvals)) / np.max([np.min(np.abs(eigvals)), 1e-10])
    return eigvals, cond


def get_covariance(Xb, offset=0.1):
    """Compute covariance matrices with eigenvalue regularization."""
    Sb = []
    for X in Xb:
        S = covariance.empirical_covariance(X, assume_centered=False)
        eigvals, cond = eig_condition_number(S)
        if np.min(eigvals) <= 1e-6:
            print(f"Adjusting covariance: min eig = {np.min(eigvals):.3e}, cond = {cond:.2f}")
            S += np.eye(S.shape[0]) * (offset - np.min(eigvals))
        Sb.append(S)
    return np.array(Sb)


##############################
# 3️⃣ Covariance Diagnostics
##############################

def upper_tri_indexing(A):
    """Return upper-triangular elements (excluding diagonal)."""
    r, c = np.triu_indices(A.shape[0], 1)
    return A[r, c]


def analyse_condition_number(df, message="", verbose=True):
    """Compute condition number of covariance matrix from DataFrame."""
    S = covariance.empirical_covariance(df, assume_centered=False)
    eigvals, cond = eig_condition_number(S)
    if verbose:
        print(f"{message}: cond = {cond:.2f}, min eig = {np.min(eigvals):.3e}, max eig = {np.max(eigvals):.3e}")
    return S, eigvals, cond


##############################
# 4️⃣ Categorical Associations
##############################

def cramers_v(x, y):
    """Cramér’s V for categorical-categorical association."""
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    if n == 0:
        return 0
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    return np.sqrt(phi2corr / denom) if denom > 0 else 0.0


def correlation_ratio(categories, measurements):
    """Correlation ratio (η) for categorical–numerical association."""
    fcat, _ = pd.factorize(categories)
    n_cat = np.max(fcat) + 1
    group_means = np.array([measurements[fcat == i].mean() for i in range(n_cat)])
    n_per_group = np.array([np.sum(fcat == i) for i in range(n_cat)])
    overall_mean = np.sum(group_means * n_per_group) / np.sum(n_per_group)
    num = np.sum(n_per_group * (group_means - overall_mean) ** 2)
    den = np.sum((measurements - overall_mean) ** 2)
    return np.sqrt(num / den) if den > 0 else 0.0


def pairwise_cov_matrix(df, dtype_map):
    """
    Compute covariance-like matrix handling categorical and numerical variables.
    - Cat–Cat: Cramér’s V
    - Cat–Num: Correlation ratio
    - Num–Num: Pearson correlation
    """
    features = df.columns
    D = len(features)
    cov = np.zeros((D, D))
    for i, fi in enumerate(features):
        for j, fj in enumerate(features):
            if j >= i:
                if dtype_map[fi] == 'c' and dtype_map[fj] == 'c':
                    cov[i, j] = cramers_v(df[fi], df[fj])
                elif dtype_map[fi] == 'c' and dtype_map[fj] == 'r':
                    cov[i, j] = correlation_ratio(df[fi], df[fj])
                elif dtype_map[fi] == 'r' and dtype_map[fj] == 'c':
                    cov[i, j] = correlation_ratio(df[fj], df[fi])
                elif dtype_map[fi] == 'r' and dtype_map[fj] == 'r':
                    cov[i, j] = pearsonr(df[fi], df[fj])[0]
                cov[j, i] = cov[i, j]
    return pd.DataFrame(cov, index=features, columns=features)


##############################
# 5️⃣ Table Preprocessing
##############################

def normalize_table(df, method="none"):
    """Normalize table columns."""
    if method == "min_max":
        return (df - df.min()) / (df.max() - df.min() + 1e-8)
    elif method == "mean":
        return (df - df.mean()) / (df.std() + 1e-8)
    else:
        return df


def process_table(
    table,
    NORM="no",
    MIN_VARIANCE=0.0,
    COND_NUM=np.inf,
    eigval_th=1e-3,
    VERBOSE=True,
):
    """Process tabular data for graph recovery models."""
    start = time()
    if VERBOSE:
        print(f"Processing input table: {table.shape}")

    # Drop all-zero rows
    table = table.loc[~(table == 0).all(axis=1)]

    # Fill NaNs with column mean
    table = table.fillna(table.mean())

    # Drop single-value columns
    single_cols = [c for c in table.columns if table[c].nunique() == 1]
    table = table.drop(columns=single_cols, errors="ignore")

    # Normalize
    table = normalize_table(table, NORM)

    # Remove duplicates
    table = table.T.drop_duplicates().T

    # Drop low variance
    low_var_cols = table.var()[table.var() < MIN_VARIANCE].index
    table = table.drop(columns=low_var_cols, errors="ignore")

    # Check conditioning
    cov_table, eigvals, cond = analyse_condition_number(table, "Processed", VERBOSE)
    itr = 1
    while cond > COND_NUM:
        if VERBOSE:
            print(f"[{itr}] Condition number too high ({cond:.2f}) – removing correlated features.")
        corr_feats = get_highly_correlated_features(cov_table)
        drop_cols = table.columns[corr_feats.astype(int)]
        table = table.drop(columns=drop_cols, errors="ignore")
        cov_table, eigvals, cond = analyse_condition_number(table, f"Iteration {itr}", VERBOSE)
        itr += 1

    if VERBOSE:
        print(f"Final processed table shape: {table.shape}")
        print(f"Processing time: {time() - start:.2f}s")

    return table


def get_highly_correlated_features(cov_matrix):
    """Identify features with high inter-correlation."""
    cov2 = covariance.empirical_covariance(cov_matrix)
    np.fill_diagonal(cov2, 0)
    vals = upper_tri_indexing(np.abs(cov2))
    threshold = np.percentile(vals, 90)
    high_pairs = np.argwhere(np.abs(cov2) >= threshold)
    counts = np.bincount(high_pairs[:, 0])
    return np.argsort(-counts)
