import torch
from torch.utils.data import DataLoader
from data_utils.data_utils import PcapDataset, MTU

class PcapDataLoader(DataLoader):
    def __init__(self, pcap_paths, labels, batch_size=64, shuffle=True, min_flow_length=2, resolution=MTU, label_mapping=None):
        """
        Initialize a DataLoader for PCAP datasets.
        
        Args:
        - pcap_paths (iterable): List of PCAP file paths.
        - labels (iterable): Corresponding labels for each PCAP file.
        - batch_size (int): Batch size for the DataLoader.
        - shuffle (bool): Whether to shuffle the data.
        - min_flow_length (int): Minimum number of packets for each flow.
        - resolution (int): Resolution of FlowPic histograms.
        - label_mapping (dict, optional): Pre-defined label mapping for consistency.
        """
        dataset = PcapDataset(
            pcap_paths=pcap_paths,
            labels=labels,
            min_flow_length=min_flow_length,
            resolution=resolution,
            label_mapping=label_mapping
        )
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)
