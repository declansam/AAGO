from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

class FireDataset(Dataset):
    def __init__(self, sequence_length=4, istrain=True):
        
        file_name = "dataset/fire_dataset.pkl"
        if not istrain:
            file_name = "dataset/fire_dataset_test.pkl"
        
        with open(file_name, "rb") as f:
            self.raw_data = pickle.load(f)
        self.sequence_length = sequence_length
        
        # Initialize scalers
        self.grid_scaler = MinMaxScaler()
        self.weather_scaler = StandardScaler()
        
        # Fit scalers and prepare sequences
        self.sequences = self._prepare_sequences()
    
    def _prepare_sequences(self):
        sequences = []
        all_grids = []
        all_weather = []
        
        # First pass: collect all data for fitting scalers
        for fire_sequence in self.raw_data:
            for grid, conditions in fire_sequence:
                all_grids.append(grid.flatten())
                all_weather.append(list(conditions.values()))
        
        # Fit scalers
        self.grid_scaler.fit(np.array(all_grids))
        self.weather_scaler.fit(np.array(all_weather))
        
        # Second pass: create scaled sequences
        for fire_sequence in self.raw_data:
            grids, weather = [], []
            for grid, conditions in fire_sequence:
                grids.append( self.grid_scaler.transform([grid.flatten()])[0] )
                weather.append(self.weather_scaler.transform([list(conditions.values())])[0])
                
            
            # Create sliding windows
            for i in range(len(grids) - self.sequence_length):
                x = np.concatenate([
                    np.array(grids[i:i+self.sequence_length]),
                    np.array(weather[i:i+self.sequence_length])
                ], axis=1)
                y = grids[i+self.sequence_length]
                sequences.append((x, y))
        
        return sequences
    
    def inverse_transform_grid(self, grid):
        return self.grid_scaler.inverse_transform(grid)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x, y = self.sequences[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)