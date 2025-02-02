import torch
from torch.utils.data import DataLoader
from dataset import FireDataset
from model import FireTransformer
from config import Config
import numpy as np
from pathlib import Path
from model import FireTransformer


def train():
    '''
    Function to train the model
    '''

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    dataset = FireDataset(Config.SEQUENCE_LENGTH)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    
    # Initialize model
    model = FireTransformer(Config).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Training loop
    best_val_loss = float('inf')
    Path('checkpoints').mkdir(exist_ok=True)

    # Main training loop
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss = 0

        # Training
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        # Print losses
        val_loss /= len(val_loader)
        print(f'Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}')

        # Save model if validation loss is lower
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')

if __name__ == "__main__":
    train()