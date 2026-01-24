import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        """Forward pass: acts as identity."""
        ctx.lambda_ = lambda_
        return x.clone()

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


def gem_1d(x: torch.Tensor, p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    GeM pooling for 1D feature maps.

    Args:
        x: Tensor of shape (B, C, L)
        p: scalar tensor (trainable or fixed) controlling the pooling sharpness
        eps: numerical stability

    Returns:
        Tensor of shape (B, C, 1)
    """
    # clamp for numerical stability, then power
    x = x.clamp(min=eps).pow(p)
    # global average pool over length
    x = F.adaptive_avg_pool1d(x, output_size=1)
    # inverse power
    return x.pow(1.0 / p)


class GeM1D(nn.Module):
    """
    GeM pooling layer for 1D CNNs.

    Adapted from https://www.kaggle.com/code/scaomath/g2net-1d-cnn-gem-pool-pytorch-train-inference

    Typical usage:
        self.pool = GeM1D(p=3.0, eps=1e-6, trainable=True)
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6, trainable: bool = True):
        super().__init__()
        self.eps = eps

        p_tensor = torch.ones(1) * p
        if trainable:
            self.p = nn.Parameter(p_tensor)
        else:
            self.register_buffer("p", p_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"GeM1D expects (B, C, L), got shape {tuple(x.shape)}")
        return gem_1d(x, self.p, self.eps)

    def __repr__(self) -> str:
        p_val = self.p.data.item() if self.p.numel() == 1 else self.p.data
        return f"{self.__class__.__name__}(p={p_val:.4f}, eps={self.eps})"


class ConfigurableMMCNN(nn.Module):
    """Multi-modal CNN from "Encrypted traffic classification: the QUIC case".

    Implements a two-branch architecture:
    - CNN branch for packet sequence statistics (pstats)
    - MLP branch for flow statistics (flowstats)

    Args:
        params: Dictionary with configuration:
            - num_classes (required): Number of output classes
            - pstats_format: "btc" (B,T,C) or "bct" (B,C,T), default "btc"
            - flowstats_dim: Dimension of flow stats, default 44
            - dropout_cnn: Dropout rate for CNN branch, default 0.1
            - dropout_flow: Dropout rate for flow branch, default 0.1
            - dropout_shared: Dropout rate for shared head, default 0.2
            - gem_p_init: Initial value for GeM pooling p parameter, default 3.0
            - gem_eps: Epsilon for GeM pooling, default 1e-6
            - lambda_rgl: DANN GRL lambda (0 to disable DANN), default 0
            - dann_fc_out_features: Hidden size for DANN domain classifier (required if lambda_rgl > 0)
            - lambda_grl_gamma: Gamma for DANN lambda scheduling (required if lambda_rgl > 0)

    Input:
        - pstats: Packet statistics tensor, shape (B, 30, 3) or (B, 3, 30) depending on pstats_format
        - flowstats: Flow statistics tensor, shape (B, flowstats_dim)

    Output:
        Dictionary containing:
        - class_preds: Classification logits (B, num_classes)
        - features: Shared embedding before classification (B, 600)
        - domain_preds: Domain logits (B, 2) if lambda_rgl > 0
    """

    def __init__(self, params):
        super(ConfigurableMMCNN, self).__init__()
        self.params = params
        self.epoch = 0

        # Extract parameters with defaults
        self.num_classes = params['num_classes']
        self.pstats_format = params.get('pstats_format', 'btc')
        self.flowstats_dim = params.get('flowstats_dim', 44)
        self.dropout_cnn = params.get('dropout_cnn', 0.1)
        self.dropout_flow = params.get('dropout_flow', 0.1)
        self.dropout_shared = params.get('dropout_shared', 0.2)
        gem_p_init = params.get('gem_p_init', 3.0)
        gem_eps = params.get('gem_eps', 1e-6)

        # CNN branch for packet statistics
        self.cnn = nn.Sequential(
            # Initial conv block
            nn.Conv1d(3, 200, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(200),

            # 3x Conv blocks (200->200)
            nn.Sequential(
                nn.Conv1d(200, 200, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(200),
            ),
            nn.Sequential(
                nn.Conv1d(200, 200, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(200),
            ),
            nn.Sequential(
                nn.Conv1d(200, 200, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(200),
            ),

            # Conv 200->300
            nn.Conv1d(200, 300, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(300),

            # Conv 300->300
            nn.Conv1d(300, 300, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(300),

            # Final conv 300->300
            nn.Conv1d(300, 300, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
        )

        # GeM pooling + BN + Dropout
        self.cnn_global_pooling = nn.Sequential(
            GeM1D(p=gem_p_init, eps=gem_eps, trainable=True),
            nn.Flatten(start_dim=1),
            nn.BatchNorm1d(300),
            nn.Dropout(p=self.dropout_cnn),
        )

        # MLP branch for flow statistics
        self.fc_flowstats = nn.Sequential(
            # Initial linear block
            nn.Linear(self.flowstats_dim, 225),
            nn.ReLU(),
            nn.BatchNorm1d(225),

            # 2x Linear blocks (225->225)
            nn.Sequential(
                nn.Linear(225, 225),
                nn.ReLU(),
                nn.BatchNorm1d(225),
            ),
            nn.Sequential(
                nn.Linear(225, 225),
                nn.ReLU(),
                nn.BatchNorm1d(225),
            ),

            # Final linear block
            nn.Linear(225, 225),
            nn.ReLU(),
            nn.BatchNorm1d(225),
            nn.Dropout(p=self.dropout_flow),
        )

        # Fusion + shared head
        # Concatenate [CNN:300, Flow:225] = 525
        self.fc_shared = nn.Sequential(
            nn.Linear(525, 600),
            nn.ReLU(),
            nn.BatchNorm1d(600),
            nn.Dropout(p=self.dropout_shared),
        )

        # Output layer
        self.out = nn.Linear(600, self.num_classes)

        # Optional DANN domain classifier
        if params.get('lambda_rgl', 0) > 0:
            assert "dann_fc_out_features" in params, "DANN requires dann_fc_out_features"
            assert "lambda_grl_gamma" in params, "DANN requires lambda_grl_gamma"

            self.domain_classifier = nn.Sequential(
                GradientReversalLayer(lambda_=params['lambda_rgl']),
                nn.Linear(600, params['dann_fc_out_features']),
                nn.ReLU(),
                nn.Linear(params['dann_fc_out_features'], 2)
            )

    def set_epoch(self, epoch):
        """Update epoch and adjust GRL lambda for DANN if enabled."""
        self.epoch = epoch
        if self.params.get('lambda_rgl', 0) > 0:
            gamma = self.params['lambda_grl_gamma']
            factor = (2 / (1 + torch.exp(torch.tensor(-gamma * epoch))) - 1).item()
            self.domain_classifier[0].lambda_ = self.params['lambda_rgl'] * factor
            print(f"Epoch {epoch}: GRL Lambda = {self.domain_classifier[0].lambda_}")

    def get_features(self, pstats, flowstats):
        """Extract shared feature embeddings before classification.

        Args:
            pstats: Packet statistics tensor
            flowstats: Flow statistics tensor

        Returns:
            Shared feature representation (B, 600)
        """
        # Handle pstats format: convert to (B, C, T) if needed
        if self.pstats_format == 'btc':
            pstats = pstats.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        # CNN branch
        out_cnn = self.cnn(pstats)
        out_cnn = self.cnn_global_pooling(out_cnn)

        # Flow stats branch
        out_flowstats = self.fc_flowstats(flowstats)

        # Concatenate and pass through shared layers
        out = torch.cat([out_cnn, out_flowstats], dim=1)
        features = self.fc_shared(out)

        return features

    def forward(self, t):
        """Forward pass.

        Args:
            t: Tuple of (pstats, flowstats)
                - pstats: shape (B, 30, 3) or (B, 3, 30) depending on pstats_format
                - flowstats: shape (B, flowstats_dim)

        Returns:
            Dictionary with:
                - class_preds: Classification logits (B, num_classes)
                - features: Shared embedding (B, 600)
                - domain_preds: Domain logits (B, 2) if DANN enabled
        """
        return_dict = dict()

        pstats, flowstats = t

        # Get shared features
        features = self.get_features(pstats, flowstats)

        # Classification
        class_preds = self.out(features)

        return_dict['class_preds'] = class_preds
        return_dict['features'] = features

        # Optional DANN domain prediction
        if self.params.get('lambda_rgl', 0) > 0:
            domain_preds = self.domain_classifier(features)
            return_dict['domain_preds'] = domain_preds

        return return_dict
