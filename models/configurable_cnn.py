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
        self.conv_type = params['conv_type']
        self.use_batch_norm = params.get('use_batch_norm', False)

        self.layers = nn.ModuleList()
        in_channels = params['input_shape']

        for conv_layer in params['conv_layers']:
            layer_type = conv_layer.get('type', self.conv_type)
            if layer_type == '1d':
                self.layers.append(nn.Conv1d(in_channels, conv_layer['out_channels'],
                                             kernel_size=conv_layer['kernel_size'],
                                             stride=conv_layer['stride'],
                                             padding=conv_layer['padding']))
                if self.use_batch_norm:
                    self.layers.append(nn.BatchNorm1d(conv_layer['out_channels']))
                self.layers.append(nn.GELU())
                self.layers.append(nn.MaxPool1d(kernel_size=params['pool_kernel_size'], stride=params['pool_stride']))
                in_channels = conv_layer['out_channels']
            elif layer_type == '2d':
                self.layers.append(nn.Conv2d(in_channels, conv_layer['out_channels'],
                                             kernel_size=conv_layer['kernel_size'],
                                             stride=conv_layer['stride'],
                                             padding=conv_layer['padding']))
                if self.use_batch_norm:
                    self.layers.append(nn.BatchNorm2d(conv_layer['out_channels']))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.MaxPool2d(kernel_size=params['pool_kernel_size'], stride=params['pool_stride']))
                in_channels = conv_layer['out_channels']
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        self.fc1 = nn.Linear(self._get_flattened_size(params['input_shape']), params['fc1_out_features'])
        self.fc2 = nn.Linear(params['fc1_out_features'], params['num_classes'])
        self.dropout = nn.Dropout(params['dropout_prob'])
        
        if params['lambda_rgl'] > 0:
            assert "dann_fc_out_features" in params, "DANN requires dann_fc_out_features"
            self.domain_classifier = nn.Sequential(
                GradientReversalLayer(lambda_=params['lambda_rgl']),
                nn.Linear(params['fc1_out_features'], params['dann_fc_out_features']),
                nn.ReLU(),
                nn.Linear(params['dann_fc_out_features'], 2)
            )

    def _get_flattened_size(self, input_shape):
        x = torch.randn(1, self.params['input_shape'], input_shape).to(next(self.parameters()).device)
        for layer in self.layers:
            x = layer(x)
        return x.numel()

    def get_features(self, x):
        """ Extracts feature embeddings before the classification layers """
        x = x.transpose(1, 2)  # Ensure correct input format
        for layer in self.layers:
            x = layer(x)
            # x = self.dropout(x)   
            
        x = x.view(x.size(0), -1)  # Flatten
        x = F.gelu(self.fc1(x))
        return x  # Return feature representation before FC layers

    def forward(self, x):
        return_dict = dict()
        
        features = self.get_features(x)
        class_preds = self.fc2(features)
        
        if self.params['lambda_rgl'] > 0:
            domain_preds = self.domain_classifier(features)
            return_dict['domain_preds'] = domain_preds
            
        return_dict['class_preds'] = class_preds
        return_dict['features'] = features
        
        # features = self.dropout(features)
        return return_dict
