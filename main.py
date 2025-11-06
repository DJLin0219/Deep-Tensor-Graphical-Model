
"""
DETECT: Deep Tensor Graphical Model Learning
---------------------------------------------
This file defines the main DETECT algorithm for precision matrix estimation
based on a differentiable D-trace loss with neural parameterization.

It includes:
- Model initialization and training (DETECT_1D)
- Loss computation
- Visualization utilities for partial correlation graphs
"""

import sys
import io
import copy
import numpy as np
import pandas as pd
from time import time
import torch
from sklearn import covariance
import matplotlib.pyplot as plt
import networkx as nx
from pyvis import network as net

from detect.detect_params import detect_params
from detect import metrics as reportMetrics
import detect.utils.prepare_data as prepare_data


##############################
# 1️⃣ DETECT_1D Wrapper
##############################

class DETECT_1D:
    """Wrapper class for DETECT algorithm (1D version), 
    following sklearn-like interface."""

    def __init__(self):
        self.covariance_ = None
        self.precision_ = None
        self.model_detect = None

    def fit(
        self,
        X,
        true_theta=None,
        eval_offset=0.1,
        centered=False,
        epochs=250,
        lr=0.002,
        INIT_DIAG=0,
        L=15,
        verbose=True
    ):
        """Fit DETECT model to the input samples X.

        Args:
            X (np.ndarray): Sample matrix of shape (n_samples, n_features).
            true_theta (np.ndarray, optional): Ground truth precision matrix for evaluation.
            eval_offset (float): Eigenvalue adjustment for ill-conditioned covariance.
            centered (bool): Whether the samples are mean-centered.
            epochs (int): Number of training epochs.
            lr (float): Learning rate for Adam optimizer.
            INIT_DIAG (int): Initialization strategy for DETECT.
            L (int): Number of unrolled iterations.
            verbose (bool): Whether to print training logs.

        Returns:
            dict: Comparison metrics between predicted and true precision matrices.
        """
        print("Running DETECT_1D...")
        start = time()

        X = prepare_data.process_table(pd.DataFrame(X), NORM='min_max', VERBOSE=verbose)
        X = np.array(X)
        M, D = X.shape
        Xb = X.reshape(1, M, D)

        true_theta_b = true_theta.reshape(1, D, D) if true_theta is not None else None

        pred_theta, compare_theta, model_detect = run_DETECT_1d_direct(
            Xb,
            trueTheta=true_theta_b,
            eval_offset=eval_offset,
            EPOCHS=epochs,
            lr=lr,
            INIT_DIAG=INIT_DIAG,
            L=L,
            VERBOSE=verbose,
        )

        self.covariance_ = covariance.empirical_covariance(X, assume_centered=centered)
        self.precision_ = pred_theta[0].detach().numpy()
        self.model_detect = model_detect
        print(f"Total runtime for DETECT_1D: {time() - start:.2f}s\n")

        return compare_theta


##############################
# 2️⃣ DETECT Core Functions
##############################

def init_DETECT_1d(lr, theta_init_offset=1.0, nF=3, H=3):
    """Initialize DETECT parameters and optimizer."""
    model = detect_params(theta_init_offset=theta_init_offset, nF=nF, H=H)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def forward_DETECT_1d(Sb, model_detect, L=15, INIT_DIAG=0, loss_Sb=None):
    """Forward pass of DETECT: predict precision matrix and compute loss."""
    from detect import detect as detect_core

    predTheta, lpen = detect_core.detect(Sb, model_detect, L=L, INIT_DIAG=INIT_DIAG)
    loss_target = Sb if loss_Sb is None else loss_Sb
    loss = loss_DETECT_1d(predTheta, loss_target, lpen)
    return predTheta, loss


def loss_DETECT_1d(theta, S, rho_val=0.01, epsilon=1e-6):
    """Differentiable D-trace Lasso loss with smooth L1 regularization."""
    B, D, _ = S.shape
    theta_sq = torch.einsum("bij,bjk->bik", theta, theta)
    term1 = 0.5 * torch.einsum("bij,bjk->bik", theta_sq, S)
    term1 = torch.einsum("bii->b", term1)
    term2 = torch.einsum("bii->b", theta)
    l1_norm = torch.sum(torch.sqrt((rho_val * theta) ** 2 + epsilon), dim=(1, 2))
    dtrace_loss = torch.sum(term1 - term2 + l1_norm) / B
    return dtrace_loss


def run_DETECT_1d_direct(
    Xb,
    trueTheta=None,
    eval_offset=0.1,
    EPOCHS=250,
    lr=0.002,
    INIT_DIAG=0,
    L=15,
    VERBOSE=True
):
    """Train DETECT directly on data."""
    Sb = prepare_data.getCovariance(Xb, offset=eval_offset)
    Xb = prepare_data.convertToTorch(Xb, req_grad=False)
    Sb = prepare_data.convertToTorch(Sb, req_grad=False)
    trueTheta = prepare_data.convertToTorch(trueTheta, req_grad=False) if trueTheta is not None else None

    model_detect, optimizer = init_DETECT_1d(lr=lr)
    PRINT_EVERY = max(1, EPOCHS // 10)

    for e in range(EPOCHS):
        optimizer.zero_grad()
        predTheta, loss = forward_DETECT_1d(Sb, model_detect, L=L, INIT_DIAG=INIT_DIAG)
        loss.backward()
        if not e % PRINT_EVERY and VERBOSE:
            print(f"[Epoch {e}/{EPOCHS}] loss = {loss.item():.4f}")
        optimizer.step()

    compare_theta = None
    if trueTheta is not None:
        compare_theta = reportMetrics.reportMetrics(
            trueTheta[0].detach().numpy(), predTheta[0].detach().numpy()
        )

    return predTheta, compare_theta, model_detect


##############################
# 3️⃣ Visualization Utilities
##############################

def get_partial_correlations(precision):
    """Compute partial correlation matrix from precision matrix."""
    precision = np.array(precision)
    D = precision.shape[0]
    rho = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            if i == j:
                rho[i, j] = 1
            else:
                rho[i, j] = -precision[i, j] / np.sqrt(precision[i, i] * precision[j, j])
    return rho


def graph_from_partial_correlations(
    rho,
    names,
    sparsity=1,
    title="",
    fig_size=12,
    PLOT=True,
    save_file=None,
    roundOFF=5,
):
    """Visualize partial correlation network."""
    G = nx.Graph()
    G.add_nodes_from(names)
    D = rho.shape[-1]

    def upper_tri(A):
        r, c = np.triu_indices(A.shape[0], 1)
        return A[r, c]

    rho_upper = np.sort(upper_tri(np.abs(rho)))
    th = rho_upper[-int(sparsity * len(rho_upper))]
    edges = []

    for i in range(D):
        for j in range(i + 1, D):
            if abs(rho[i, j]) > th:
                color = "green" if rho[i, j] > 0 else "red"
                G.add_edge(names[i], names[j], color=color, weight=round(rho[i, j], roundOFF))
                edges.append((names[i], names[j], round(rho[i, j], roundOFF), color))

    if PLOT:
        fig = plt.figure(figsize=(fig_size, fig_size))
        pos = nx.spring_layout(G)
        edge_colors = [G.edges[e]["color"] for e in G.edges]
        nx.draw(G, pos, with_labels=True, edge_color=edge_colors)
        plt.title(title)
        if save_file:
            plt.savefig(save_file, bbox_inches="tight")
        plt.close(fig)

    return G, edges


def viz_graph_from_precision(theta, feature_names, sparsity=0.1, title=""):
    """Generate interactive graph from precision matrix."""
    rho = get_partial_correlations(theta)
    G, _ = graph_from_partial_correlations(rho, feature_names, sparsity=sparsity)
    print(f"Graph with {len(G.nodes())} nodes and {len(G.edges())} edges.")
    Gv = net.Network(notebook=True, height="750px", width="100%", heading=title)
    Gv.from_nx(G)
    return G, Gv
