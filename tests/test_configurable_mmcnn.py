"""Basic tests for ConfigurableMMCNN to verify coherence with the paper architecture."""
import torch
import pytest
from models.configurable_mmcnn import ConfigurableMMCNN, GeM1D, GradientReversalLayer


# --- Fixtures ---

@pytest.fixture
def base_params():
    return {'num_classes': 102, 'flowstats_dim': 44}


@pytest.fixture
def dann_params():
    return {
        'num_classes': 102,
        'flowstats_dim': 44,
        'lambda_rgl': 1.0,
        'dann_fc_out_features': 256,
        'lambda_grl_gamma': 10.0,
    }


@pytest.fixture
def batch():
    """Standard batch: pstats (B,30,3) in 'btc' format, flowstats (B,44)."""
    B = 8
    pstats = torch.randn(B, 30, 3)
    flowstats = torch.randn(B, 44)
    return pstats, flowstats


# --- Shape Tests ---

class TestOutputShapes:
    def test_class_preds_shape(self, base_params, batch):
        model = ConfigurableMMCNN(base_params)
        model.eval()
        out = model(batch)
        assert out['class_preds'].shape == (8, 102)

    def test_features_shape(self, base_params, batch):
        model = ConfigurableMMCNN(base_params)
        model.eval()
        out = model(batch)
        assert out['features'].shape == (8, 600)

    def test_no_domain_preds_without_dann(self, base_params, batch):
        model = ConfigurableMMCNN(base_params)
        model.eval()
        out = model(batch)
        assert 'domain_preds' not in out

    def test_domain_preds_shape_with_dann(self, dann_params, batch):
        model = ConfigurableMMCNN(dann_params)
        model.eval()
        out = model(batch)
        assert out['domain_preds'].shape == (8, 2)

    def test_variable_batch_size(self, base_params):
        model = ConfigurableMMCNN(base_params)
        model.eval()
        for B in [1, 4, 16, 32]:
            pstats = torch.randn(B, 30, 3)
            flowstats = torch.randn(B, 44)
            out = model((pstats, flowstats))
            assert out['class_preds'].shape == (B, 102)
            assert out['features'].shape == (B, 600)

    def test_different_num_classes(self):
        for nc in [5, 10, 50, 200]:
            params = {'num_classes': nc, 'flowstats_dim': 44}
            model = ConfigurableMMCNN(params)
            model.eval()
            pstats = torch.randn(2, 30, 3)
            flowstats = torch.randn(2, 44)
            out = model((pstats, flowstats))
            assert out['class_preds'].shape == (2, nc)


# --- CNN Spatial Dimension Tests ---

class TestCNNSpatialDimensions:
    """Verify that the CNN reduces (B, 3, 30) -> (B, 300, 10) as expected."""

    def test_cnn_output_spatial_dim(self, base_params):
        model = ConfigurableMMCNN(base_params)
        model.eval()
        x = torch.randn(4, 3, 30)  # (B, C, T) format
        out = model.cnn(x)
        # Paper: input length 30 -> output length 10
        # 30 (same pad) -> 30 -> 30 -> 30 -> (30-5+1)=26 -> (26-5+1)=22 -> (22-4)/2+1=10
        assert out.shape == (4, 300, 10)

    def test_cnn_channel_progression(self, base_params):
        model = ConfigurableMMCNN(base_params)
        model.eval()
        x = torch.randn(2, 3, 30)

        # After initial conv block (layers 0,1,2): channels = 200
        out = model.cnn[:3](x)
        assert out.shape[1] == 200

        # After 3 residual-style blocks (layers 3,4,5): still 200
        out = model.cnn[:6](x)
        assert out.shape[1] == 200

        # After first 200->300 conv (layers 6,7,8): channels = 300
        out = model.cnn[:9](x)
        assert out.shape[1] == 300


# --- PSTATS Format Tests ---

class TestPstatsFormat:
    def test_btc_format(self, base_params):
        """Default 'btc' format: input (B, T=30, C=3)."""
        model = ConfigurableMMCNN(base_params)
        model.eval()
        pstats = torch.randn(4, 30, 3)  # (B, T, C)
        flowstats = torch.randn(4, 44)
        out = model((pstats, flowstats))
        assert out['class_preds'].shape == (4, 102)

    def test_bct_format(self):
        """'bct' format: input (B, C=3, T=30), no transpose needed."""
        params = {'num_classes': 102, 'flowstats_dim': 44, 'pstats_format': 'bct'}
        model = ConfigurableMMCNN(params)
        model.eval()
        pstats = torch.randn(4, 3, 30)  # (B, C, T)
        flowstats = torch.randn(4, 44)
        out = model((pstats, flowstats))
        assert out['class_preds'].shape == (4, 102)

    def test_btc_and_bct_give_same_result(self, base_params):
        """Same data in different formats should give the same output."""
        torch.manual_seed(42)
        pstats_btc = torch.randn(4, 30, 3)
        pstats_bct = pstats_btc.transpose(1, 2)  # (B, 3, 30)
        flowstats = torch.randn(4, 44)

        model_btc = ConfigurableMMCNN({**base_params, 'pstats_format': 'btc'})
        model_bct = ConfigurableMMCNN({**base_params, 'pstats_format': 'bct'})
        # Copy weights
        model_bct.load_state_dict(model_btc.state_dict())
        model_btc.eval()
        model_bct.eval()

        out_btc = model_btc((pstats_btc, flowstats))
        out_bct = model_bct((pstats_bct, flowstats))
        assert torch.allclose(out_btc['class_preds'], out_bct['class_preds'], atol=1e-5)


# --- GeM Pooling Tests ---

class TestGeM:
    def test_gem_output_shape(self):
        gem = GeM1D(p=3.0, eps=1e-6, trainable=True)
        x = torch.rand(4, 300, 10) + 0.1  # positive values
        out = gem(x)
        assert out.shape == (4, 300, 1)

    def test_gem_with_p1_is_avg_pool(self):
        """When p=1, GeM should be equivalent to average pooling."""
        gem = GeM1D(p=1.0, eps=1e-6, trainable=False)
        x = torch.rand(4, 64, 10) + 0.1
        out_gem = gem(x).squeeze(-1)
        out_avg = x.mean(dim=2)
        assert torch.allclose(out_gem, out_avg, atol=1e-5)

    def test_gem_trainable_parameter(self):
        gem = GeM1D(p=3.0, trainable=True)
        assert isinstance(gem.p, torch.nn.Parameter)
        assert gem.p.requires_grad

    def test_gem_non_trainable(self):
        gem = GeM1D(p=3.0, trainable=False)
        assert not isinstance(gem.p, torch.nn.Parameter)

    def test_gem_rejects_wrong_dims(self):
        gem = GeM1D()
        with pytest.raises(ValueError):
            gem(torch.randn(4, 300))  # 2D instead of 3D


# --- GRL Tests ---

class TestGRL:
    def test_grl_forward_is_identity(self):
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(4, 64)
        out = grl(x)
        assert torch.allclose(out, x)

    def test_grl_reverses_gradient(self):
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(4, 64, requires_grad=True)
        out = grl(x)
        loss = out.sum()
        loss.backward()
        # Gradient should be -1 * ones (reversed)
        assert torch.allclose(x.grad, -torch.ones_like(x))

    def test_grl_lambda_scaling(self):
        grl = GradientReversalLayer(lambda_=2.0)
        x = torch.randn(4, 64, requires_grad=True)
        out = grl(x)
        loss = out.sum()
        loss.backward()
        assert torch.allclose(x.grad, -2.0 * torch.ones_like(x))


# --- DANN Integration Tests ---

class TestDANN:
    def test_epoch_updates_grl_lambda(self):
        params = {
            'num_classes': 102,
            'flowstats_dim': 44,
            'lambda_rgl': 1.0,
            'dann_fc_out_features': 256,
            'lambda_grl_gamma': 0.1,  # Small gamma so schedule doesn't saturate
        }
        model = ConfigurableMMCNN(params)
        model.set_epoch(0)
        lambda_at_0 = model.domain_classifier[0].lambda_
        model.set_epoch(5)
        lambda_at_5 = model.domain_classifier[0].lambda_
        # At epoch 0, factor=0; at epoch 5 with gamma=0.1, factor≈0.46
        assert lambda_at_0 == 0.0
        assert 0 < lambda_at_5 < 1.0

    def test_epoch_zero_grl_lambda_is_zero(self, dann_params):
        model = ConfigurableMMCNN(dann_params)
        model.set_epoch(0)
        # At epoch 0, factor = 2/(1+exp(0)) - 1 = 2/2 - 1 = 0
        assert model.domain_classifier[0].lambda_ == 0.0

    def test_dann_requires_fc_out_features(self):
        params = {'num_classes': 102, 'lambda_rgl': 1.0, 'lambda_grl_gamma': 10.0}
        with pytest.raises(AssertionError, match="dann_fc_out_features"):
            ConfigurableMMCNN(params)

    def test_dann_requires_gamma(self):
        params = {'num_classes': 102, 'lambda_rgl': 1.0, 'dann_fc_out_features': 256}
        with pytest.raises(AssertionError, match="lambda_grl_gamma"):
            ConfigurableMMCNN(params)


# --- Feature Extraction Tests ---

class TestGetFeatures:
    def test_get_features_shape(self, base_params, batch):
        model = ConfigurableMMCNN(base_params)
        model.eval()
        pstats, flowstats = batch
        features = model.get_features(pstats, flowstats)
        assert features.shape == (8, 600)

    def test_get_features_matches_forward(self, base_params, batch):
        model = ConfigurableMMCNN(base_params)
        model.eval()
        pstats, flowstats = batch
        features = model.get_features(pstats, flowstats)
        out = model(batch)
        assert torch.allclose(features, out['features'], atol=1e-5)


# --- Parameter Count Test ---

class TestParameterCount:
    def test_approximate_param_count(self, base_params):
        """Paper states 2.2M trainable parameters for 102 classes."""
        model = ConfigurableMMCNN(base_params)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Should be approximately 2.2M (allow some tolerance for BN params etc.)
        assert 2_000_000 < total_params < 2_500_000, f"Got {total_params} params"


# --- Gradient Flow Tests ---

class TestGradientFlow:
    def test_gradients_flow_to_cnn(self, base_params, batch):
        model = ConfigurableMMCNN(base_params)
        model.train()
        out = model(batch)
        loss = out['class_preds'].sum()
        loss.backward()
        # Check first conv layer has gradients
        first_conv = model.cnn[0]
        assert first_conv.weight.grad is not None
        assert first_conv.weight.grad.abs().sum() > 0

    def test_gradients_flow_to_flowstats(self, base_params, batch):
        model = ConfigurableMMCNN(base_params)
        model.train()
        out = model(batch)
        loss = out['class_preds'].sum()
        loss.backward()
        first_linear = model.fc_flowstats[0]
        assert first_linear.weight.grad is not None
        assert first_linear.weight.grad.abs().sum() > 0

    def test_dann_gradient_reversal(self, dann_params, batch):
        """Domain classifier loss should produce reversed gradients in feature extractor."""
        model = ConfigurableMMCNN(dann_params)
        model.set_epoch(10)  # Ensure non-zero lambda
        model.train()
        out = model(batch)

        # Only backprop domain loss
        domain_loss = out['domain_preds'].sum()
        domain_loss.backward()

        # The shared layer should have gradients (reversed by GRL)
        assert model.fc_shared[0].weight.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
