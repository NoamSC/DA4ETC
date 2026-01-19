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


class ConfigurableCNN(nn.Module):
    def __init__(self, params):
        super(ConfigurableCNN, self).__init__()
        self.params = params
        self.epoch = 0
        self.conv_type = params['conv_type']
        self.use_batch_norm = params.get('use_batch_norm', False)

        # Build feature extractor (fully convolutional)
        self.feature_extractor = nn.ModuleList()
        in_channels = params['input_shape']

        feature_config = params['feature_extractor']
        for conv_layer in feature_config['conv_layers']:
            in_channels = self._add_conv_block(
                self.feature_extractor,
                in_channels,
                conv_layer,
                feature_config['pool_kernel_size'],
                feature_config['pool_stride']
            )

        # Build label predictor (convs + FCs)
        self.label_predictor_convs = nn.ModuleList()
        label_config = params['label_predictor']

        for conv_layer in label_config['conv_layers']:
            in_channels = self._add_conv_block(
                self.label_predictor_convs,
                in_channels,
                conv_layer,
                label_config['pool_kernel_size'],
                label_config['pool_stride']
            )

        # Label predictor FC layers
        self.label_predictor_fcs = nn.ModuleList()
        flattened_size = self._get_flattened_size(params['input_shape'],
                                                    self.feature_extractor,
                                                    self.label_predictor_convs)

        fc_in_features = flattened_size
        for fc_out_features in label_config['fc_layers']:
            self.label_predictor_fcs.append(nn.Linear(fc_in_features, fc_out_features))
            fc_in_features = fc_out_features

        # Final classification layer
        self.label_output = nn.Linear(fc_in_features, params['num_classes'])
        self.dropout = nn.Dropout(params['dropout_prob'])

        # Build domain classifier if DANN is enabled
        if params['lambda_rgl'] > 0:
            assert "lambda_grl_gamma" in params, "DANN requires lambda_grl_gamma"
            assert 'domain_classifier' in params, "DANN requires domain_classifier config"

            self.domain_classifier_convs = nn.ModuleList()
            domain_config = params['domain_classifier']

            # Use same input channels as after feature extractor
            domain_in_channels = params['feature_extractor']['conv_layers'][-1]['out_channels']

            for conv_layer in domain_config['conv_layers']:
                domain_in_channels = self._add_conv_block(
                    self.domain_classifier_convs,
                    domain_in_channels,
                    conv_layer,
                    domain_config['pool_kernel_size'],
                    domain_config['pool_stride']
                )

            # Domain classifier FC layers
            self.domain_classifier_fcs = nn.ModuleList()
            domain_flattened_size = self._get_flattened_size(params['input_shape'],
                                                              self.feature_extractor,
                                                              self.domain_classifier_convs)

            # Add GRL at the beginning
            self.grl = GradientReversalLayer(lambda_=params['lambda_rgl'])

            fc_in_features = domain_flattened_size
            for fc_out_features in domain_config['fc_layers']:
                self.domain_classifier_fcs.append(nn.Linear(fc_in_features, fc_out_features))
                fc_in_features = fc_out_features

            # Final domain classification layer (2 classes: source/target)
            self.domain_output = nn.Linear(fc_in_features, 2)

    def _add_conv_block(self, module_list, in_channels, conv_layer, pool_kernel_size, pool_stride):
        """Helper to add a conv block (Conv + BN + GELU + Pool) to a ModuleList."""
        layer_type = conv_layer.get('type', self.conv_type)

        if layer_type == '1d':
            module_list.append(nn.Conv1d(in_channels, conv_layer['out_channels'],
                                         kernel_size=conv_layer['kernel_size'],
                                         stride=conv_layer['stride'],
                                         padding=conv_layer['padding']))
            if self.use_batch_norm:
                module_list.append(nn.BatchNorm1d(conv_layer['out_channels']))
            module_list.append(nn.GELU())
            module_list.append(nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride))
        elif layer_type == '2d':
            module_list.append(nn.Conv2d(in_channels, conv_layer['out_channels'],
                                         kernel_size=conv_layer['kernel_size'],
                                         stride=conv_layer['stride'],
                                         padding=conv_layer['padding']))
            if self.use_batch_norm:
                module_list.append(nn.BatchNorm2d(conv_layer['out_channels']))
            module_list.append(nn.GELU())
            module_list.append(nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride))
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        return conv_layer['out_channels']
            
    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.params['lambda_rgl'] > 0:
            gamma = self.params['lambda_grl_gamma']
            factor = (2 / (1 + torch.exp(torch.tensor(-gamma * epoch))) - 1)
            self.grl.lambda_ = self.params['lambda_rgl'] * factor
            print(f"Epoch {epoch}: GRL Lambda = {self.grl.lambda_}")

    def _get_flattened_size(self, input_shape, *conv_modules):
        """Calculate flattened size after passing through multiple conv module lists."""
        x = torch.randn(1, self.params['input_shape'], input_shape)
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu'
        x = x.to(device)

        for module_list in conv_modules:
            for layer in module_list:
                x = layer(x)
        return x.numel()

    def get_features(self, x):
        """Extracts feature embeddings from the feature extractor (shared backbone)."""
        x = x.transpose(1, 2)  # Ensure correct input format

        # Pass through feature extractor
        for layer in self.feature_extractor:
            x = layer(x)

        return x  # Return feature representation after feature extractor

    def forward(self, x):
        return_dict = dict()

        # Extract shared features
        features = self.get_features(x)

        # Label prediction branch
        label_features = features
        for layer in self.label_predictor_convs:
            label_features = layer(label_features)

        label_features = label_features.view(label_features.size(0), -1)  # Flatten
        label_features = self.dropout(label_features)

        for fc_layer in self.label_predictor_fcs:
            label_features = F.gelu(fc_layer(label_features))
            label_features = self.dropout(label_features)

        class_preds = self.label_output(label_features)

        # Domain prediction branch (if DANN is enabled)
        if self.params['lambda_rgl'] > 0:
            # Apply GRL to shared features
            domain_features = self.grl(features)

            # Pass through domain classifier convs
            for layer in self.domain_classifier_convs:
                domain_features = layer(domain_features)

            domain_features = domain_features.view(domain_features.size(0), -1)  # Flatten
            domain_features = self.dropout(domain_features)

            for fc_layer in self.domain_classifier_fcs:
                domain_features = F.gelu(fc_layer(domain_features))
                domain_features = self.dropout(domain_features)

            domain_preds = self.domain_output(domain_features)
            return_dict['domain_preds'] = domain_preds

        return_dict['class_preds'] = class_preds
        return_dict['features'] = label_features

        return return_dict
