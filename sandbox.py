from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

# Assuming PcapFlowPicDataset and QUICFlowPicDataset are already defined
from FlowPic.dataset import PcapFlowPicDataset, QUICFlowPicDataset
from FlowPic.model import FlowPicCNN

# Updated Label Mapping based on the new categories
LABEL_MAPPING = {
    "Streaming Media": [
        ("vpnnonvpn", ["Netflix", "Vimeo", "YouTube"]),
        ("quic", ["YouTube", "Google Music"])
    ],
    "Remote Access": [
        ("vpnnonvpn", ["RDP", "SSH"]),
        ("quic", [])
    ],
    "File Transfer": [
        ("vpnnonvpn", ["Rsync", "SCP", "SFTP"]),
        ("quic", ["Google Drive"])
    ],
    "Chat/VoIP": [
        ("vpnnonvpn", ["Skype-Chat", "VoIP"]),
        ("quic", [])
    ],
    "Web Applications": [
        ("vpnnonvpn", []),
        ("quic", ["Google Doc"])
    ],
    "Web Browsing": [
        ("vpnnonvpn", []),
        ("quic", ["Google Search"])
    ]
}

# Load datasets
def create_dataloaders(pcap_dir, quic_dir, batch_size=16, cache_dir=None):
    vpn_dataset = PcapFlowPicDataset(pcap_dir, cache_dir=cache_dir)
    quic_dataset = QUICFlowPicDataset(quic_dir, cache_dir=quic_dir / 'flowpic_cache')

    # Remove samples with no category label
    vpn_indices = [i for i, y in enumerate(vpn_dataset.ys) if y is not None]
    vpn_dataset.Xs = [vpn_dataset.Xs[i] for i in vpn_indices]
    vpn_dataset.ys = [vpn_dataset.ys[i] for i in vpn_indices]

    quic_indices = [i for i, y in enumerate(quic_dataset.ys) if y is not None]
    quic_dataset.Xs = [quic_dataset.Xs[i] for i in quic_indices]
    quic_dataset.ys = [quic_dataset.ys[i] for i in quic_indices]

    # Split datasets into train and validation
    train_size_vpn = int(0.7 * len(vpn_dataset))
    val_size_vpn = len(vpn_dataset) - train_size_vpn
    train_size_quic = int(0.7 * len(quic_dataset))
    val_size_quic = len(quic_dataset) - train_size_quic

    vpn_train, vpn_val = torch.utils.data.random_split(vpn_dataset, [train_size_vpn, val_size_vpn])
    quic_train, quic_val = torch.utils.data.random_split(quic_dataset, [train_size_quic, val_size_quic])

    # Data loaders
    vpn_train_loader = DataLoader(vpn_train, batch_size=batch_size, shuffle=True)
    vpn_val_loader = DataLoader(vpn_val, batch_size=batch_size, shuffle=False)
    quic_train_loader = DataLoader(quic_train, batch_size=batch_size, shuffle=True)
    quic_val_loader = DataLoader(quic_val, batch_size=batch_size, shuffle=False)

    return vpn_train_loader, vpn_val_loader, quic_train_loader, quic_val_loader

# Model and Training
def train_model(train_loader, val_loader, params, num_epochs=10, lr=1e-3, device='cpu'):
    model = FlowPicCNN(params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels.argmax(dim=1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.argmax(dim=1)).sum().item()
            total += labels.size(0)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/total:.4f}, Accuracy: {correct/total:.4f}')

    return model

# Evaluation
def evaluate_model(model, data_loader, device='cpu'):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs.unsqueeze(1))
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.argmax(dim=1).cpu().numpy())

    return accuracy_score(y_true, y_pred)

# Training and evaluation script
def main():
    pcap_dir = Path("data/vpnnonvpn")  # Replace with your PCAP dataset directory
    quic_dir = Path("data/QUIC/pretraining")  # Replace with your QUIC dataset directory

    # Load datasets
    vpn_train_loader, vpn_val_loader, quic_train_loader, quic_val_loader = create_dataloaders(pcap_dir, quic_dir, batch_size=16)

    # Model parameters (shared for both datasets)
    model_params = {
        'num_classes': len(LABEL_MAPPING),   # Aligned number of common labels
        'image_size': 1500,
        'conv_layers': [
            {'in_channels': 1, 'out_channels': 32, 'kernel_size': 5, 'stride': 2, 'padding': 2},
            {'in_channels': 32, 'out_channels': 64, 'kernel_size': 5, 'stride': 2, 'padding': 2},
            {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        ],
        'pool_kernel_size': 2,
        'pool_stride': 2,
        'fc1_out_features': 512,
        'dropout_prob': 0.3
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train models
    print("Training VPN-nonVPN model...")
    vpn_model = train_model(vpn_train_loader, vpn_val_loader, model_params, device=device, num_epochs=10)
    
    print("Training QUIC model...")
    quic_model = train_model(quic_train_loader, quic_val_loader, model_params, device=device, num_epochs=10)

    # Evaluate each model on each dataset
    print("\nEvaluating VPN-nonVPN model on both datasets...")
    vpn_model_vpn_train_acc = evaluate_model(vpn_model, vpn_train_loader, device=device)
    vpn_model_vpn_val_acc = evaluate_model(vpn_model, vpn_val_loader, device=device)
    vpn_model_quic_train_acc = evaluate_model(vpn_model, quic_train_loader, device=device)
    vpn_model_quic_val_acc = evaluate_model(vpn_model, quic_val_loader, device=device)

    print("\nEvaluating QUIC model on both datasets...")
    quic_model_vpn_train_acc = evaluate_model(quic_model, vpn_train_loader, device=device)
    quic_model_vpn_val_acc = evaluate_model(quic_model, vpn_val_loader, device=device)
    quic_model_quic_train_acc = evaluate_model(quic_model, quic_train_loader, device=device)
    quic_model_quic_val_acc = evaluate_model(quic_model, quic_val_loader, device=device)

    # Print results
    print(f"\nVPN-nonVPN model on VPN Train: {vpn_model_vpn_train_acc:.4f}")
    print(f"VPN-nonVPN model on VPN Val: {vpn_model_vpn_val_acc:.4f}")
    print(f"VPN-nonVPN model on QUIC Train: {vpn_model_quic_train_acc:.4f}")
    print(f"VPN-nonVPN model on QUIC Val: {vpn_model_quic_val_acc:.4f}")

    print(f"\nQUIC model on VPN Train: {quic_model_vpn_train_acc:.4f}")
    print(f"QUIC model on VPN Val: {quic_model_vpn_val_acc:.4f}")
    print(f"QUIC model on QUIC Train: {quic_model_quic_train_acc:.4f}")
    print(f"QUIC model on QUIC Val: {quic_model_quic_val_acc:.4f}")

if __name__ == "__main__":
    main()

Amazon Prime Video
Amazon Shopping
AMC
BBC News
Bible
Bitmoji
BuzzFeed
CNN
Color Ballz
Colorfy
Dancing Line
Domino's Pizza USA
Dune!
eBay
Epocrates
ESPN
Fire Rides
Fitbit
Flashlight
Flipagram
FollowMyHealth
FOX News
Fruit Ninja
Gmail
GoodRx
Google Chrome
Google Docs
Google Drive
Google Maps
Google Photos
Google Play Books
Google Translate
Google Search
Groupon
Grubhub
Hotspot Shield VPN
Layout from Instagram
letgo
Live Wallpapers Now
Lose It!
Marco Polo
McDonald's
Medscape
Merriam-Webster Dictionary
MyChart
MyFitnessPal
MyRadar Weather Radar
National Geographic
Newsroom
NFL
Nike Run Club
OpenTable
Pandora Music
PayPal
Peak - Brain Training
People Magazine
Perfect365
Photomath
Pinterest
Pocket Pool
Puzzledom
Recolor
Rolly Vortex
Sarahah
Scanner App
Secret Apps
Shazam
Snake VS Block
SoundCloud
Spotify
Starbucks
tbh
TED
Text Free
The New York Times
The Wall Street Journal
Tinder
Trivia Crack
USA Today
Walmart
Waze
Wish
Yarn - Chat Stories
Yelp
YouTube
Zillow
