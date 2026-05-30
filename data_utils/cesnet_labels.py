"""Label-mapping helpers for the CESNET-TLS-Year22 dataset.

Lives in the library (data_utils) so scripts can import it without depending on the
training entrypoint. Previously defined in train_per_week_cesnet.py.
"""

import json


def load_label_mapping(dataset_root):
    """Load label mapping from a CESNET dataset root.

    Returns (label_indices_mapping, num_classes), where label_indices_mapping maps
    app_name (str, as found in the parquet APP column) -> contiguous label index.
    """
    label_mapping_path = dataset_root / 'label_mapping.json'
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)

    app_names = sorted(label_mapping.keys())
    label_indices_mapping = {app_name: i for i, app_name in enumerate(app_names)}
    return label_indices_mapping, len(app_names)
