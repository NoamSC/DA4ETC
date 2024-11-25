import torch
from torch.utils.data import DataLoader
from data_utils.data_utils import PcapDataset, MTU

class PcapDataLoader(DataLoader):
    def __init__(self, pcap_paths=None, labels=None, dataset=None, batch_size=64, shuffle=True, min_flow_length=2, resolution=MTU, label_mapping=None):
        """
        Initialize a DataLoader for PCAP datasets.

        Args:
        - pcap_paths (iterable, optional): List of PCAP file paths.
        - labels (iterable, optional): Corresponding labels for each PCAP file.
        - dataset (PcapDataset, optional): Pre-initialized dataset to use directly.
        - batch_size (int): Batch size for the DataLoader.
        - shuffle (bool): Whether to shuffle the data.
        - min_flow_length (int): Minimum number of packets for each flow.
        - resolution (int): Resolution of FlowPic histograms.
        - label_mapping (dict, optional): Pre-defined label mapping for consistency.

        Raises:
        - ValueError: If neither `pcap_paths` nor `dataset` is provided.
        """
        if dataset is None:
            if pcap_paths is None or labels is None:
                raise ValueError("Either `pcap_paths` and `labels` or `dataset` must be provided.")
            dataset = PcapDataset(
                pcap_paths=pcap_paths,
                labels=labels,
                min_flow_length=min_flow_length,
                resolution=resolution,
                label_mapping=label_mapping
            )

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)
