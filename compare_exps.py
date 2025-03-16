from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


exps_path = Path('exps')
result_matrix_filename = 'cross_domain_accuracy_matrix.npy'

num_locations = 6
locations = [
    'AwsCont', 'BenContainer',  'CabSpicy1',
    'HujiPC', 'TLVunContainer1', 'TLVunContainer2'
]
baseline_name = 'CNN_mmd2_0'
exps_names = [
    # 'CNN_mmd2_1e-1',
    'CNN_mmd2_1e1_v2',
]

baseline_results = np.load(exps_path / baseline_name / result_matrix_filename)
baseline_delta = 1 - baseline_results

for exp_name in exps_names:
    print(exp_name)
    exp_path = exps_path / exp_name
    assert exp_path.exists()
    exp_results = np.load(exp_path / result_matrix_filename)
    deltas = (exp_results - baseline_results) / baseline_delta
    print(deltas)
    plt.figure(figsize=(8, 8))
    plt.imshow(deltas, cmap='viridis', interpolation='none', vmin=0, vmax=1)
    plt.colorbar(label='Accuracy')
    for i in range(num_locations):
        for j in range(num_locations):
            plt.text(j, i, f'{deltas[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)
    plt.xticks(ticks=np.arange(num_locations), labels=locations, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(num_locations), labels=locations)
    plt.title('Cross-Domain Accuracy Matrix')
    plt.xlabel('Test Domain')
    plt.ylabel('Train Domain')
    plt.tight_layout()
    plt.show()
    plt.savefig(exp_name + '_delta_accuracy_matrix.png', dpi=300)
