from pathlib import Path

import numpy as np


exps_path = Path('exps')
result_matrix_filename = 'cross_domain_accuracy_matrix.npy'

baseline_name = 'fat_CNN_advanced_mmd_features_mmd_0'
exps_names = [
    'fat_CNN_advanced_mmd_features_mmd_1e-1',
    'fat_CNN_advanced_mmd_features_mmd_1e1',
]

baseline_results = np.load(exps_path / baseline_name / result_matrix_filename)
baseline_delta = 1 - baseline_results

for exp_name in exps_names:
    print(exp_name)
    exp_path = exps_path / exp_name
    assert exp_path.exists()
    exp_results = np.load(exp_path / result_matrix_filename)
    print((exp_results - baseline_results) / baseline_delta)
