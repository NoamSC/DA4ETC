import random
import numpy as np
import hashlib
from bayes_opt import BayesianOptimization
from config import Config
from train_model_on_different_locations import run_full_exp

seed = 42
# Define architectures as a list of possible configurations
# ARCHITECTURE_CHOICES = [
#     # # Base architecture
#     # {
#     #     'conv_type': '1d',
#     #     'input_shape': 256,
#     #     'conv_layers': [
#     #         {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
#     #         {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
#     #         {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
#     #         {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
#     #         {'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1}
#     #     ],
#     #     'pool_kernel_size': 2,
#     #     'pool_stride': 2,
#     #     'fc1_out_features': 64,
#     #     'dropout_prob': 0.3,
#     #     'use_batch_norm': True
#     # },
#     # # Deeper architecture
#     # {
#         # 'conv_type': '1d',
#         # 'input_shape': 256,
#         # 'conv_layers': [
#         #     {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
#         #     {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
#         #     {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
#         #     {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
#         #     {'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
#         #     {'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1}
#         # ],
#         # 'pool_kernel_size': 2,
#         # 'pool_stride': 2,
#         # 'fc1_out_features': 128,
#         # 'dropout_prob': 0.4,
#         # 'use_batch_norm': True
#     # },
#     # Wide architecture
#     {
#         'conv_type': '1d',
#         'input_shape': 256,
#         'conv_layers': [
#             {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
#             {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
#             {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
#             {'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
#             {'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1}
#         ],
#         'pool_kernel_size': 2,
#         'pool_stride': 2,
#         'fc1_out_features': 128,
#         'dropout_prob': 0.3,
#         'use_batch_norm': True
#     },
#     # Shallow architecture
#     {
#         'conv_type': '1d',
#         'input_shape': 256,
#         'conv_layers': [
#             {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
#             {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
#             {'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1}
#         ],
#         'pool_kernel_size': 2,
#         'pool_stride': 2,
#         'fc1_out_features': 64,
#         'dropout_prob': 0.3,
#         'use_batch_norm': True
#     }
# ]
ARCHITECTURE_CHOICES = [
    # Narrow architecture
    {
        'conv_type': '1d',
        'input_shape': 256,
        'conv_layers': [
            {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ],
        'pool_kernel_size': 2,
        'pool_stride': 2,
        'fc1_out_features': 64,
        'dropout_prob': 0.2,
        'use_batch_norm': True
    },
    # Reduced depth architecture
    {
        'conv_type': '1d',
        'input_shape': 256,
        'conv_layers': [
            {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ],
        'pool_kernel_size': 2,
        'pool_stride': 2,
        'fc1_out_features': 64,
        'dropout_prob': 0.25,
        'use_batch_norm': True
    },
    # Light architecture
    {
        'conv_type': '1d',
        'input_shape': 256,
        'conv_layers': [
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ],
        'pool_kernel_size': 2,
        'pool_stride': 2,
        'fc1_out_features': 32,
        'dropout_prob': 0.2,
        'use_batch_norm': True
    },
    # Very shallow architecture
    {
        'conv_type': '1d',
        'input_shape': 256,
        'conv_layers': [
            {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ],
        'pool_kernel_size': 2,
        'pool_stride': 2,
        'fc1_out_features': 32,
        'dropout_prob': 0.15,
        'use_batch_norm': True
    }
]


# Architecture mapping function
def get_architecture(index):
    return ARCHITECTURE_CHOICES[int(round(index))]

# Generate unique experiment name
def generate_exp_name(lambda_mmd, architecture_idx, mmd_bandwidth):
    param_str = f"{round(lambda_mmd, 5)}_{architecture_idx}_{round(mmd_bandwidth, 5)}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    return f"bayesian_search_mmd_v3_{param_hash}"

# Black-box function to optimize
def black_box_function(lambda_mmd, architecture_idx, mmd_bandwidth):
    experiment_name = generate_exp_name(lambda_mmd, architecture_idx, mmd_bandwidth)
    
    config = Config(
        EXPERIMENT_NAME=experiment_name,
        LAMBDA_MMD=10 ** lambda_mmd, # changing from logarithmic
        MODEL_PARAMS=get_architecture(architecture_idx),
        MMD_BANDWIDTHS=[10 ** mmd_bandwidth],
        SEED=seed
    )
    
    # Run actual experiment
    accuracy = run_full_exp(config)
    
    result_str = f"{experiment_name} | Lambda_MMD={lambda_mmd}, Architecture={architecture_idx}, MMD_Bandwidth={mmd_bandwidth} --> Accuracy: {accuracy}\n"
    print(f"Testing Config: {result_str}")
    
    # Log results to file
    with open("bayesian_search_results_v3.txt", "a") as f:
        f.write(result_str)
    
    return accuracy

# Define hyperparameter search space
pbounds = {
    "lambda_mmd": (-5, 0),
    "architecture_idx": (0, len(ARCHITECTURE_CHOICES) - 1),
    "mmd_bandwidth": (-5, 1),
}

# Run Bayesian Optimization
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=seed,
)

optimizer.maximize(
    init_points=10,  # Number of initial random evaluations
    n_iter=100,      # Number of optimization iterations
)

# Log best result
best_result = f"Best found configuration: {optimizer.max}\n"
print(best_result)
with open("bayesian_search_results_v3.txt", "a") as f:
    f.write(best_result)
