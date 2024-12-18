import os
import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd

MTU = 2048  # Maximum Transmission Unit

def session_2d_histogram(ts, sizes, resolution=MTU, max_delta_time=10):
    """
    Create a 2D histogram from time and size data with reduced resolution.
    
    Args:
    - ts (array): Timestamps of the packets.
    - sizes (array): Packet sizes.
    - resolution (int): Resolution of the output FlowPic (default is MTU).
    - max_delta_time (float): Time axis length [secs] of the FlowPic.
    
    Returns:
    - H (array): 2D histogram of size [resolution x resolution].
    """
    if not max_delta_time:
        max_delta_time = ts[-1] - ts[0]

    ts_norm = ((np.array(ts) - ts[0]) / max_delta_time) * MTU
    
    # Create bins for both size and time, controlling the resolution
    bin_edges = np.linspace(0, MTU, resolution + 1)  # Creating bins of desired resolution
    
    # Generate the 2D histogram
    H, xedges, yedges = np.histogram2d(sizes, ts_norm, bins=(bin_edges, bin_edges))
    
    return H.astype(np.uint16)

def sessions_to_flowpic(sessions, resolution=MTU):
    """
    Convert all sessions in a given dictionary into FlowPic format.
    """
    dataset = []
    for session_key, session in sessions.items():
        ts = np.array(session['time_deltas'])
        sizes = np.array(session['sizes'])

        h = session_2d_histogram(ts, sizes, resolution=resolution)
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


class PcapDataset(Dataset):
    def __init__(self, pcap_paths, labels, min_flow_length=2, resolution=MTU, label_mapping=None):
        """
        Initialize the dataset with PCAP paths and labels, enforcing a consistent label mapping.

        Args:
        - pcap_paths (iterable): Iterable of PCAP file paths.
        - labels (iterable): Corresponding labels for each PCAP file.
        - min_flow_length (int): Minimum number of packets for each flow.
        - resolution (int): Resolution of FlowPic histograms.
        - label_mapping (dict, optional): Pre-defined label mapping for consistency.
        """
        self.Xs = []
        self.ys = []
        
        # Use pre-defined label mapping if provided; otherwise, create one dynamically
        self.label_mapping = label_mapping if label_mapping is not None else {}
        self.label_counter = len(self.label_mapping)

        # Process each PCAP file
        for pcap_path, label in tqdm(zip(pcap_paths, labels), total=len(pcap_paths)):
            # Add new labels to the mapping dynamically if not already present
            if label not in self.label_mapping:
                self.label_mapping[label] = self.label_counter
                self.label_counter += 1

            sessions = parse_pcap(pcap_path, min_flow_length=min_flow_length)
            flowpics = sessions_to_flowpic(sessions, resolution=resolution)
            
            for flowpic in flowpics:
                self.Xs.append(flowpic)
                self.ys.append(self.label_mapping[label])

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        flowpic = np.array(self.Xs[idx], dtype=np.float32)
        label = np.array(self.ys[idx], dtype=np.float32)
        
        return torch.Tensor(flowpic), torch.Tensor(label)

def extract_pcap_info(path):
    """
    Extract metadata (location, date, app, vpn type) from PCAP file path.

    Args:
    - path (str): Path to the PCAP file.

    Returns:
    - Tuple (location, date, app, vpn_type).
    """
    parts = Path(path).parts
    location, date, app, vpn_type = parts[3], pd.to_datetime(parts[4], format='%Y%m%d_%H%M%S'), parts[5], parts[6]
    return location, date, app, vpn_type
