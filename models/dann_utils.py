import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        """Forward pass: acts as identity."""
        ctx.lambda_ = lambda_
        return x.view_as(x)  # Identity operation that preserves computation graph

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: reverses gradient by multiplying with -lambda_."""
        lambda_ = ctx.lambda_
        grad_input = -lambda_ * grad_output
        return grad_input, None  # None because lambda_ is not trainable


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


def compute_grl_lambda(epoch: int, lambda_rgl: float, gamma: float) -> float:
    """Compute the GRL lambda based on training progress.

    Uses the schedule from the DANN paper:
    lambda = lambda_rgl * (2 / (1 + exp(-gamma * epoch)) - 1)

    Args:
        epoch: Current epoch number
        lambda_rgl: Base lambda value
        gamma: Scaling factor for the schedule

    Returns:
        Scheduled lambda value
    """
    factor = (2 / (1 + torch.exp(torch.tensor(-gamma * epoch))) - 1)
    return lambda_rgl * factor.item()
