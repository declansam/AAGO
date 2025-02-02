import torch
import torch.nn as nn
import math
from config import Config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self attention block
        att_out, _ = self.attn(x, x, x)
        x = x + self.dropout(att_out)
        x = self.norm1(x)
        
        # Feed forward block
        ff_out = self.ffn(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x

class FireTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(config.GRID_SIZE + config.WEATHER_FEATURES)
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.GRID_SIZE + config.WEATHER_FEATURES, config.D_MODEL),
            nn.GELU(),
            nn.Dropout(config.DROPOUT)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.D_MODEL)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.D_MODEL, config.N_HEAD, config.DROPOUT)
            for _ in range(config.N_LAYERS)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.D_MODEL, config.D_MODEL // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.D_MODEL // 2, config.GRID_SIZE),
            nn.Sigmoid()  # Bound outputs between 0-1
        )
    
    def forward(self, x):
        # Scale inputs to [0-1]
        grid = x[..., :Config.GRID_SIZE]
        weather = x[..., Config.GRID_SIZE:]
        x = torch.cat([grid, weather], dim=-1)
        
        # Input normalization
        x = self.input_norm(x)
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Process through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global pooling across sequence dimension
        x = x.mean(dim=1)
        
        # Output projection and scaling
        x = self.output_proj(x)
        
        # Scale back to [0-255]
        return x