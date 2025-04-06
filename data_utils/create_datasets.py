import pickle
from multiprocessing import Pool
import pandas as pd
from pathlib import Path

from data_utils.data_utils import extract_pcap_info, PcapDataset

# Example configuration object (replace with your actual configuration)
class DataConfig:
    TRAIN_SPLIT_RATIO = 0.7
    MIN_FLOW_LENGTH = 100
    RESOLUTION = 256

cfg = DataConfig()

pcaps = pd.read_csv('pcap_paths.csv').values[:, 1]

# Extract metadata from PCAP paths
df = pd.DataFrame(
    [(str(p), *extract_pcap_info(p)) for p in pcaps],
    columns=['pcap_path', 'location', 'date', 'app', 'vpn_type']
)

# Function to process each location
def process_location(location, label_mapping):
    try:
        print(f"Processing location: {location}")

        # Filter and sample data
        df_location = df[df.location == location].sort_values(by='date')

        # Split into train/validation
        split_index = int(len(df_location) * cfg.TRAIN_SPLIT_RATIO)
        df_train, df_val = df_location[:split_index], df_location[split_index:]

        # Create datasets
        train_dataset = PcapDataset(
            pcap_paths=df_train['pcap_path'].tolist(),
            labels=df_train['app'].tolist(),
            min_flow_length=cfg.MIN_FLOW_LENGTH,
            resolution=cfg.RESOLUTION,
            label_mapping=label_mapping
        )
        val_dataset = PcapDataset(
            pcap_paths=df_val['pcap_path'].tolist(),
            labels=df_val['app'].tolist(),
            min_flow_length=cfg.MIN_FLOW_LENGTH,
            resolution=cfg.RESOLUTION,
            label_mapping=label_mapping
        )

        # Save datasets to pickle
        filename = Path('data/ben_bucket/cached_datasets') / f'datasets_{location}_{int(cfg.RESOLUTION)}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump((train_dataset, val_dataset), f)

        print(f"Processing for {location} completed. Output saved to {filename}.")
    except Exception as e:
        print(f"Error processing location {location}: {e}")

if __name__ == "__main__":
    # List of unique locations
    df = df[df.app.isin(['Amazon', 'Google Search', 'Twitch', 'Youtube'])]
    label_mapping = {'Amazon': 0, 'Google Search': 1, 'Twitch': 2, 'Youtube': 3}

    locations = df['location'].unique()
    locations = [location for location in locations if df[df.location == location]['app'].count() > 1000]

    # Use multiprocessing to process each location in parallel
    for location in locations:
        process_location(location, label_mapping)
