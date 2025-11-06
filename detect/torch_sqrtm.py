import torch
from torch.autograd import Function


class MatrixSquareRoot(Function):
    """
    Differentiable matrix square root using Newton–Schulz iteration.

    Computes the matrix square root for a positive definite matrix A such that:
        sqrtm(A) @ sqrtm(A) ≈ A

    This implementation supports autograd (backpropagation) and is numerically
    stable for well-conditioned, positive definite matrices.

    Note:
        - The operation is not defined for singular matrices (with zero eigenvalues).
        - This function uses a fixed number of Newton–Schulz iterations.
    """

    @staticmethod
    def forward(ctx, input, num_iters: int = 10):
        """
        Forward pass: compute matrix square root via Newton–Schulz iteration.

        Args:
            ctx: Autograd context for saving tensors for backward.
            input (torch.Tensor): Input positive definite matrix of shape (D, D).
            num_iters (int, optional): Number of Newton–Schulz iterations. Default = 10.

        Returns:
            torch.Tensor: The matrix square root of the input (same shape as input).
        """
        dim = input.shape[-1]
        norm = torch.norm(input)
        Y = input / norm
        I = torch.eye(dim, device=input.device, dtype=input.dtype)
        Z = I.clone()

        for _ in range(num_iters):
            T = 0.5 * (3.0 * I - Z @ Y)
            Y = Y @ T
            Z = T @ Z

        sqrtm = Y * torch.sqrt(norm)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradient w.r.t input using Lyapunov equation.

        Args:
            ctx: Autograd context from forward pass.
            grad_output (torch.Tensor): Gradient from upstream.

        Returns:
            torch.Tensor: Gradient of the loss w.r.t input.
        """
        sqrtm, = ctx.saved_tensors
        grad_input = 0.5 * torch.linalg.solve(
            sqrtm.T, grad_output + grad_output.T
        )
        return grad_input, None


# Expose as callable function
torch_sqrtm = MatrixSquareRoot.apply


if __name__ == "__main__":
    # Simple numerical check
    A = torch.randn(5, 5)
    A = A.T @ A  # make it positive definite
    A.requires_grad = True

    sqrtA = torch_sqrtm(A)
    print("Input A:\n", A)
    print("sqrtm(A):\n", sqrtA)
    print("Check reconstruction error: ||A - sqrtm(A)^2|| =",
          torch.norm(A - sqrtA @ sqrtA).item())

    # Gradient check
    from torch.autograd import gradcheck
    A_double = A.double().requires_grad_(True)
    print("Gradcheck:", gradcheck(MatrixSquareRoot.apply, (A_double,)))
