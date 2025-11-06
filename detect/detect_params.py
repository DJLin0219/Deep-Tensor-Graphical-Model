import torch
import torch.nn as nn


class DetectParams(torch.nn.Module):
    """
    DETECT parameterization module.

    This class defines the learnable hyperparameters used in DETECT,
    including the debiasing threshold function (rho), the adaptive regularization
    parameter (lambda), and the initialization offset (theta_init_offset).

    The module parameterizes these quantities as small neural networks
    that are jointly optimized during training via backpropagation.
    """

    def __init__(self, theta_init_offset, nF, H, USE_CUDA=False):
        """
        Initialize the DETECT parameter networks.

        Args:
            theta_init_offset (float): Initial eigenvalue offset (should be > 0.1 for stability).
            nF (int): Number of input features for entrywise thresholding.
            H (int): Hidden layer size for the internal neural networks.
            USE_CUDA (bool): Use GPU if True, else CPU.
        """
        super(DetectParams, self).__init__()
        self.dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
        self.theta_init_offset = nn.Parameter(
            torch.Tensor([theta_init_offset]).type(self.dtype)
        )
        self.nF = nF
        self.H = H
        self.rho_l1 = self._build_rhoNN()
        self.lambda_f = self._build_lambdaNN()
        self.zero = torch.Tensor([0]).type(self.dtype)

    def _build_rhoNN(self):
        """Neural network for learning the element-wise threshold (rho)."""
        l1 = nn.Linear(self.nF, self.H).type(self.dtype)
        lH1 = nn.Linear(self.H, self.H).type(self.dtype)
        l2 = nn.Linear(self.H, 1).type(self.dtype)
        return nn.Sequential(
            l1, nn.Tanh(),
            lH1, nn.Tanh(),
            l2, nn.Softplus()
        ).type(self.dtype)

    def _build_lambdaNN(self):
        """Neural network for adaptive regularization parameter (lambda)."""
        l1 = nn.Linear(2, self.H).type(self.dtype)
        l2 = nn.Linear(self.H, 1).type(self.dtype)
        return nn.Sequential(
            l1, nn.Tanh(),
            l2, nn.Softplus()
        ).type(self.dtype)

    def eta_forward(self, X, S, Lambda, lamb, k, F3=[]):
        """
        Perform bias-corrected soft-thresholding update for theta.

        Args:
            X (torch.Tensor): Intermediate precision matrix (batch, D, D)
            S (torch.Tensor): Covariance matrix (batch, D, D)
            Lambda (torch.Tensor): Dual variable (batch, D, D)
            lamb (float): Current regularization parameter
            k (int): Iteration index
            F3 (list, optional): Optional additional feature inputs.

        Returns:
            tuple:
                theta_new (torch.Tensor): Updated precision estimate.
                rho_val (torch.Tensor): Element-wise threshold values.
        """
        batch_size, shape1, shape2 = X.shape

        # Reshape and build feature vectors
        Xr = X.reshape(batch_size, -1, 1)
        Sr = S.reshape(batch_size, -1, 1)
        feature_vector = torch.cat((Xr, Sr), -1)

        if len(F3) > 0:
            F3r = F3.reshape(batch_size, -1, 1)
            feature_vector = torch.cat((feature_vector, F3r), -1)

        # Elementwise threshold computation
        rho_val = self.rho_l1(feature_vector).reshape(X.shape)

        # Bias-corrected shrinkage update
        theta_new = torch.sign(X) * torch.max(
            self.zero, torch.abs(X + Lambda / lamb) - rho_val
        )

        return theta_new, (rho_val * lamb)

    def lambda_forward(self, normF, prev_lambda, k=0):
        """
        Adaptive update of the regularization parameter lambda.

        Args:
            normF (float): Frobenius norm difference from the previous iteration.
            prev_lambda (float): Previous lambda value.
            k (int): Iteration index.

        Returns:
            torch.Tensor: Updated lambda value.
        """
        feature_vector = torch.Tensor([normF, prev_lambda]).type(self.dtype)
        return self.lambda_f(feature_vector)
