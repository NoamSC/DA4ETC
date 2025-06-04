import random
from pathlib import Path
import os
import json

import torch
import numpy as np
import hashlib
from bayes_opt import BayesianOptimization
from config import Config
# from train_model_on_different_locations import run_full_exp
from simple_model_train import run_full_exp

search_name = 'allot_dann_bsearch_v6'
base_experiment_path = Path("exps") / search_name
cfg = Config(BASE_EXPERIMENTS_PATH=base_experiment_path, SEED=42)
    
# Generate unique experiment name
def generate_exp_name(lambda_mmd, mmd_bandwidth):
    param_str = f"{round(lambda_mmd, 5)}_{round(mmd_bandwidth, 5)}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    return f"{param_hash}"

# Black-box function to optimize
def black_box_function(lambda_rgl, LAMBDA_DANN):
    experiment_name = generate_exp_name(lambda_rgl, LAMBDA_DANN)
    model_params = cfg.MODEL_PARAMS.copy()
    model_params['lambda_rgl'] = 10 ** lambda_rgl
    cfg.EXPERIMENT_NAME = experiment_name
    cfg.LAMBDA_DANN = 10 ** LAMBDA_DANN
    cfg.MODEL_PARAMS = model_params

    assert cfg.MODEL_PARAMS['lambda_rgl'] == 10 ** lambda_rgl
    assert cfg.LAMBDA_DANN == 10 ** LAMBDA_DANN
    
    # Run actual experiment
    accuracy = run_full_exp(cfg)
    
    result_str = f"{experiment_name} | lambda_rgl={lambda_rgl}, LAMBDA_DANN={LAMBDA_DANN} --> Accuracy: {accuracy}\n"
    print(f"Testing Config: {result_str}")
    
    # Log results to file
    with open(f"{search_name}.txt", "a") as f:
        f.write(result_str)
    
    return accuracy

def get_already_tested():
    already_done = {}
    exp_dir = Path('exps/allot_dann_bsearch_v5')
    for train_log_file in exp_dir.glob('*/plots/training_history.pth'):
        train_log = torch.load(train_log_file, weights_only=False)
        max_accuracy = np.max(train_log['val_accuracies'])
        
        json_path = train_log_file.parent.parent / 'config.json'
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        lambda_rgl = config_dict['MODEL_PARAMS']['lambda_rgl']
        lambda_dann = config_dict['LAMBDA_DANN']
        already_done[(lambda_rgl, lambda_dann)] = max_accuracy
    
    return already_done


# Define hyperparameter search space
pbounds = {
    "lambda_rgl": (-6, 1),
    # "architecture_idx": (0, len(ARCHITECTURE_CHOICES) - 1),
    "LAMBDA_DANN": (-6, 1),
}

# Run Bayesian Optimization
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=cfg.SEED,
)

# Register previous points
for (lambda_rgl, LAMBDA_DANN), acc in get_already_tested().items():
    optimizer.register(
        params={"lambda_rgl": lambda_rgl, "LAMBDA_DANN": LAMBDA_DANN},
        target=acc,
    )

optimizer.maximize(
    init_points=10,  # Number of initial random evaluations
    n_iter=100000,      # Number of optimization iterations
)

# Log best result
best_result = f"Best found configuration: {optimizer.max}\n"
print(best_result)
with open(f"{search_name}.txt", "a") as f:
    f.write(best_result)
