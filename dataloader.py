import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scapy.all import rdpcap, IP, TCP, UDP
import socket
from tqdm import tqdm

MTU = 1500  # Maximum Transmission Unit (packet size limit)
DELTA_T = 60  # Time interval for splitting sessions
TPS = 60  # Time per session
MIN_TPS = 50  # Minimum time per session to consider

# List of allowed genres for cross-platform apps
cross_platform_by_genre = ['Books & Reference', 'Auto & Vehicles', 'Business', 'Communication', 
                           'Dating', 'House & Home', 'Lifestyle', 'Medical', 'Music & Audio', 
                           'News & Magazines', 'Shopping']

# Create a mapping from genre to numerical labels
label_mapping = {genre: idx for idx, genre in enumerate(cross_platform_by_genre)}

def session_2d_histogram(ts, sizes):
    # ts_norm = map(int, ((np.array(ts) - ts[0]) / (ts[-1] - ts[0])) * MTU)
    ts_norm = ((np.array(ts) - ts[0]) / (ts[-1] - ts[0])) * MTU
    H, xedges, yedges = np.histogram2d(sizes, ts_norm, bins=(range(0, MTU + 1, 1), range(0, MTU + 1, 1)))
    return H.astype(np.uint16)

def sessions_to_flowpic(sessions):
    """
    Convert all sessions in a given dictionary into FlowPic format.
    """
    dataset = []
    for session_key, session in sessions.items():
        ts = np.array(session['time_deltas'])
        sizes = np.array(session['sizes'])

        if len(ts) > 1:  # Consider sessions with at least 2 packets
            h = session_2d_histogram(ts, sizes)
            dataset.append(h)

    return dataset


def parse_pcap(pcap_path, min_flow_length=2):
    """
    Parse a pcap file assuming raw IP packets.
    Each session is a dictionary with 'start_time', 'time_deltas', and 'sizes'.

    Args:
    - pcap_path (str): Path to the PCAP file.
    - min_flow_length (int): Minimum number of packets required for a session to be considered.
    """
    sessions = {}
    packets = rdpcap(pcap_path)
    
    for packet in packets:
        if IP not in packet:  # Ensure it's an IP packet
            continue

        ip = packet[IP]  # Extract the IP layer
        proto = packet[IP].payload  # Extract the transport layer (TCP/UDP)

        if not (proto.haslayer(TCP) or proto.haslayer(UDP)):  # Ensure it's TCP or UDP
            continue

        # Extract source and destination IPs and ports
        src_ip = ip.src
        dst_ip = ip.dst
        sport = proto.sport if proto.haslayer(TCP) or proto.haslayer(UDP) else None
        dport = proto.dport if proto.haslayer(TCP) or proto.haslayer(UDP) else None

        # Use a tuple of source IP, destination IP, source port, and destination port as the session key
        session_key = (src_ip, sport, dst_ip, dport)
        
        # Initialize the session if it's new
        if session_key not in sessions:
            sessions[session_key] = {
                'start_time': packet.time,  # Start timestamp of the session
                'time_deltas': [],  # List to hold time deltas
                'sizes': []  # List to hold packet sizes
            }

        # Update the session data with the current packet's info
        session = sessions[session_key]
        size = len(packet)  # Get packet size
        session['time_deltas'].append(packet.time - session['start_time'])  # Append time delta
        session['sizes'].append(size)  # Append packet size

    # Filter out sessions that don't meet the min_flow_length criterion
    filtered_sessions = {k: v for k, v in sessions.items() if len(v['time_deltas']) >= min_flow_length}
    
    return filtered_sessions

def temporal_split(sessions, split_ratio=0.8):
    """
    Split the session data into train and test sets based on the global timestamp percentiles.
    
    Args:
    - sessions (dict): Parsed session data from `parse_pcap`.
    - split_ratio (float): Ratio for the train-test split (default is 0.8).
    
    Returns:
    - train_sessions (dict): Sessions for the training set.
    - test_sessions (dict): Sessions for the testing set.
    """
    all_timestamps = []

    # Collect all timestamps from all sessions
    for session_key, (start_ts, ts_list, sizes) in sessions.items():
        all_timestamps.extend(np.float32(ts_list))
    
    # Compute the split time at the given split_ratio (e.g., 80th percentile)
    split_time = np.percentile(all_timestamps, split_ratio * 100)

    train_sessions = {}
    test_sessions = {}

    # Split sessions based on the split_time
    for session_key, (start_ts, ts_list, sizes) in sessions.items():
        train_ts = []
        train_sizes = []
        test_ts = []
        test_sizes = []

        for ts, size in zip(ts_list, sizes):
            if ts <= split_time:
                train_ts.append(ts)
                train_sizes.append(size)
            else:
                test_ts.append(ts)
                test_sizes.append(size)

        if train_ts:  # Only add if there's data in this split
            train_sessions[session_key] = (start_ts, train_ts, train_sizes)
        if test_ts:  # Only add if there's data in this split
            test_sessions[session_key] = (start_ts, test_ts, test_sizes)

    return train_sessions, test_sessions


class PcapFlowPicDataset(Dataset):
    def __init__(self, pcap_paths, labels, split_ratio=0.8, train_mode=True):
        """
        Initialize the dataset with PCAP paths and labels, applying a train-test split.
        
        Args:
        - pcap_paths (list): List of PCAP file paths.
        - labels (list): Corresponding labels for each PCAP file.
        - split_ratio (float): Train-test split ratio (default is 0.8).
        - train_mode (bool): If True, return training set. If False, return test set.
        """
        self.train_mode = train_mode
        self.Xs_train = []
        self.ys_train = []
        self.Xs_test = []
        self.ys_test = []
        
        # Process each PCAP file
        for pcap_path, label in tqdm(zip(pcap_paths, labels)):
            sessions = parse_pcap(pcap_path)
            train_sessions, test_sessions = temporal_split(sessions, split_ratio)

            # Process training data
            flowpics_train = sessions_to_flowpic(train_sessions)
            for flowpic in flowpics_train:
                self.Xs_train.append(flowpic)
                self.ys_train.append(label_mapping[label])

            # Process testing data
            flowpics_test = sessions_to_flowpic(test_sessions)
            for flowpic in flowpics_test:
                self.Xs_test.append(flowpic)
                self.ys_test.append(label_mapping[label])

    def __len__(self):
        """
        Returns the total number of FlowPic samples in either the training or test dataset.
        """
        if self.train_mode:
            return len(self.Xs_train)
        else:
            return len(self.Xs_test)

    def __getitem__(self, idx):
        """
        Returns a FlowPic matrix and its corresponding label.
        
        Args:
        - idx (int): Index of the sample.
        
        Returns:
        - tuple: A tuple (FlowPic, label).
        """
        if self.train_mode:
            flowpic = np.array(self.Xs_train[idx], dtype=np.float32)
            label = np.array(self.ys_train[idx], dtype=np.float32)
        else:
            flowpic = np.array(self.Xs_test[idx], dtype=np.float32)
            label = np.array(self.ys_test[idx], dtype=np.float32)
        
        return torch.Tensor(flowpic), torch.Tensor(label)



def create_dataloader(pcap_paths, labels, batch_size=64, shuffle=True, split_ratio=0.8):
    """
    Creates two DataLoaders, one for training and one for testing, using a temporal split.
    
    Args:
    - pcap_paths (list): List of PCAP file paths.
    - labels (list): Corresponding labels for each PCAP file.
    - batch_size (int): Batch size for the DataLoader.
    - shuffle (bool): Whether to shuffle the training data.
    - split_ratio (float): Train-test split ratio (default is 0.8).
    
    Returns:
    - train_loader (DataLoader): A PyTorch DataLoader for the training dataset.
    - test_loader (DataLoader): A PyTorch DataLoader for the testing dataset.
    """
    # Create training dataset and DataLoader
    train_dataset = PcapFlowPicDataset(pcap_paths, labels, split_ratio=split_ratio, train_mode=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # Create testing dataset and DataLoader
    test_dataset = PcapFlowPicDataset(pcap_paths, labels, split_ratio=split_ratio, train_mode=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle test data
    
    return train_loader, test_loader

