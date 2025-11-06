import torch

def get_optimizers(model_detect, lr_detect=0.002, use_optimizer='adam'):
    """
    Initialize the optimizer for the DETECT model.

    Args:
        model_detect (torch.nn.Module): The DETECT model instance.
        lr_detect (float, optional): Learning rate for the optimizer. Default is 0.002.
        use_optimizer (str, optional): The optimizer type. Default is 'adam'.

    Returns:
        torch.optim.Optimizer: The initialized optimizer for DETECT.
    """
    if use_optimizer == 'adam':
        optimizer_detect = torch.optim.Adam(
            model_detect.parameters(),
            lr=lr_detect,
            betas=(0.9, 0.999),
            eps=1e-08
        )
    else:
        raise ValueError("Unsupported optimizer type. Please use 'adam'.")

    return optimizer_detect


def batch_matrix_sqrt(A):
    """
    Compute the matrix square root for a single or a batch of positive semi-definite (PSD) matrices.

    Args:
        A (torch.Tensor): Input matrix or batch of matrices. 
                          Shape can be (D, D) for a single matrix or (B, D, D) for a batch.

    Returns:
        torch.Tensor: The matrix square roots, same shape as input.
    """
    if A.dim() == 2:
        return torch_sqrtm(A)

    n = A.shape[0]
    sqrtm_torch = torch.zeros_like(A)
    for i in range(n):
        sqrtm_torch[i] = torch_sqrtm(A[i])
    return sqrtm_torch


def get_frobenius_norm(A, single=False):
    """
    Compute the Frobenius norm for a single matrix or a batch of matrices.

    Args:
        A (torch.Tensor): Input matrix or batch of matrices.
                          Shape (D, D) for single or (B, D, D) for batch.
        single (bool, optional): If True, compute norm for single matrix only.
                                 If False, compute mean norm over batch. Default is False.

    Returns:
        torch.Tensor: Frobenius norm (scalar).
    """
    if single:
        return torch.sum(A ** 2)
    return torch.mean(torch.sum(A ** 2, dim=(1, 2)))


def detect(Sb, model, lambda_init=1, L=15, INIT_DIAG=0, USE_CUDA=False):
    """
    DETECT: Debiased Estimation of Tensor-based Conditional Topology.

    This function implements the iterative unrolled optimization procedure 
    for estimating the precision (inverse covariance) matrices under 
    the DETECT framework. It follows an alternating minimization structure 
    similar to ADMM, incorporating bias-corrected updates.

    Args:
        Sb (torch.Tensor): Sample covariance matrices. 
                           Shape (batch, D, D) or (D, D) for single matrix.
        model (torch.nn.Module): The DETECT model instance (contains lambda_forward, eta_forward, etc.).
        lambda_init (float, optional): Initial regularization coefficient. Default = 1.
        L (int, optional): Number of unrolled iterations. Default = 15.
        INIT_DIAG (int, optional): Initialization mode for theta.
                                   0 - full covariance inverse, 
                                   1 - diagonal inverse. Default = 0.
        USE_CUDA (bool, optional): Whether to use GPU. Default = False.

    Returns:
        tuple:
            theta_pred (torch.Tensor): Estimated precision matrices (batch, D, D).
            rho_val (float): Final penalty parameter value.
    """
    D = Sb.shape[-1]
    if Sb.dim() == 2:
        Sb = Sb.reshape(1, D, D)

    # === Initialization ===
    if INIT_DIAG == 1:
        batch_diags = 1 / (torch.diagonal(Sb, dim1=-2, dim2=-1) + model.theta_init_offset)
        theta_init = torch.diag_embed(batch_diags)
    else:
        theta_init = torch.inverse(
            Sb + model.theta_init_offset * torch.eye(D).expand_as(Sb).type_as(Sb)
        )

    theta_pred = theta_init
    identity_mat = torch.eye(D).expand_as(Sb)
    if USE_CUDA:
        identity_mat = identity_mat.cuda()

    # Initialize Lagrange multiplier (Lambda)
    Lambda = torch.randn_like(theta_pred) * 0.01
    zero = torch.Tensor([0])
    dtype = torch.FloatTensor
    if USE_CUDA:
        zero = zero.cuda()
        dtype = torch.cuda.FloatTensor
        Lambda = Lambda.cuda()

    # Initialize lambda via model
    lambda_k = model.lambda_forward(zero + lambda_init, zero, k=0)

    # === Iterative updates ===
    for k in range(L):
        # Step 1: Compute Theta^{(k+1)} via eigendecomposition
        A = Sb + lambda_k * identity_mat
        B = identity_mat + lambda_k * theta_pred - Lambda

        eigvals, eigvecs = torch.linalg.eigh(A)
        B_tilde = torch.matmul(eigvecs.transpose(-1, -2), torch.matmul(B, eigvecs))

        sigma_i = eigvals.unsqueeze(-1)
        sigma_j = eigvals.unsqueeze(-2)
        C = 2 / (sigma_i + sigma_j)
        G_tilde = B_tilde * C
        theta_k1 = torch.matmul(eigvecs, torch.matmul(G_tilde, eigvecs.transpose(-1, -2)))

        # Step 2: Update theta via bias-corrected eta_forward
        theta_pred, rho_val = model.eta_forward(theta_k1, Sb, Lambda, lambda_k, k, theta_pred)

        # Step 3: Update dual variable (Lambda)
        Lambda = Lambda + lambda_k * (theta_k1 - theta_pred)

        # Step 4: Update lambda via model (adaptive regularization)
        lambda_k = model.lambda_forward(
            torch.Tensor([get_frobenius_norm(theta_pred - theta_k1)]).type(dtype),
            lambda_k,
            k
        )

    return theta_pred, rho_val
