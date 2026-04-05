import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os

# --- Model Definition (Mandatory for loading weights) ---

class StackedBiLSTMTransformerModel(nn.Module):
    """
    A hybrid model combining a stacked Bidirectional LSTM for local temporal
    feature extraction followed by a Transformer Encoder for global sequence modeling.
    (Extracted from user's training script)
    """
    def __init__(self, 
                 input_dim: int, 
                 num_classes: int, 
                 hidden_dim: int = 384, 
                 nhead: int = 8, 
                 num_lstm_layers: int = 2,
                 num_transformer_layers: int = 1):
        
        super().__init__()
        
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.GELU(), 
            nn.Dropout(0.1),
        )

        # Max sequence length 256, matches original design
        self.positional_encoding = nn.Parameter(torch.zeros(1, 256, hidden_dim))

        lstm_dropout = 0.1 if num_lstm_layers > 1 else 0.0
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim // 2,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout, 
            bidirectional=True
        )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 4,
            dropout=0.1, 
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer, 
            num_layers=num_transformer_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256), 
            nn.LayerNorm(256),
            nn.GELU(), 
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # NOTE: Weights init is skipped here as we load a checkpoint, 
        # but the structure must be identical.

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        
        features = self.frame_encoder(x)
        
        if T > self.positional_encoding.shape[1]:
             # This check is kept for robustness
             raise ValueError(f"input sequence length ({T}) exceeds max positional encoding length ({self.positional_encoding.shape[1]})")
        features = features + self.positional_encoding[:, :T, :]
        
        lstm_out, _ = self.temporal_lstm(features)
        
        transformer_out = self.transformer_encoder(
            lstm_out, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Masked Average Pooling
        if src_key_padding_mask is not None:
            mask = ~src_key_padding_mask.unsqueeze(-1)
            masked_output = transformer_out * mask
            summed = torch.sum(masked_output, dim=1)
            count = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / count
        else:
            pooled = torch.mean(transformer_out, dim=1) 
        
        return self.classifier(pooled)


# --- PreProcessor Class for API Input ---

class PreProcessor:
    """
    Handles feature extraction, temporal differencing, normalization, and 
    resampling/padding for a single sequence of raw landmark data, 
    mimicking the logic from FixedLandmarkDataset.
    """
    def __init__(self, stats_path, label_map_path, max_frames=70):
        with open(stats_path, 'r') as f: stats = json.load(f)
        with open(label_map_path, 'r') as f: self.full_label_map = json.load(f)

        self.max_frames = max_frames
        self.spatial_dim = 1742
        self.input_dim = self.spatial_dim * 2
        
        # Load global normalization stats
        self.mean = torch.tensor(stats['spatial_mean'] + stats['temporal_mean'], dtype=torch.float32)
        self.std = torch.tensor(stats['spatial_std'] + stats['temporal_std'], dtype=torch.float32)
        self.std[self.std < 1e-6] = 1.0

        # Create index-to-gloss map (assuming the top 200 classes were trained)
        all_glosses = sorted(self.full_label_map.keys(), key=lambda g: self.full_label_map[g])
        selected_glosses = all_glosses[:200]
        self.idx_to_gloss = {i: gloss for i, gloss in enumerate(selected_glosses)}

    def _extract_spatial_features(self, frame):
        """Extracts and pads/truncates features for a single frame."""
        if not isinstance(frame, dict): return None
        # Must match the features used in FixedLandmarkDataset: 
        # pose, left_hand, right_hand, face, left_hand_engineered, right_hand_engineered
        
        def safe_get(key, size):
            data = np.array(frame.get(key, []), dtype=np.float32).flatten()
            if len(data) > size: data = data[:size]
            elif len(data) < size: data = np.pad(data, (0, size - len(data)))
            return data
        
        return np.concatenate([
            safe_get('pose', 132), safe_get('left_hand', 84), safe_get('right_hand', 84),
            safe_get('face', 1404), safe_get('left_hand_engineered', 19), safe_get('right_hand_engineered', 19)
        ])

    def preprocess(self, raw_frames):
        """
        Takes a list of raw landmark frames (dictionaries) and returns a 
        normalized, resampled PyTorch tensor ready for the model.
        """
        if not isinstance(raw_frames, list) or len(raw_frames) < 5: 
            raise ValueError("input sequence too short or invalid format.")

        # 1. Extract spatial features
        spatial_features_list = [self._extract_spatial_features(frame) for frame in raw_frames]
        if any(f is None for f in spatial_features_list): 
            raise ValueError("could not extract spatial features from all frames.")

        spatial = np.array(spatial_features_list, dtype=np.float32)
        
        if np.isnan(spatial).any() or np.isinf(spatial).any(): 
            raise ValueError("raw features contain NaN or Inf values.")
            
        # 2. Calculate temporal difference features
        temporal = np.diff(spatial, axis=0, prepend=spatial[0:1])
        features = np.concatenate([spatial, temporal], axis=1)

        # 3. Resample/truncate to max_frames (70)
        if len(features) != self.max_frames:
             indices = np.linspace(0, len(features)-1, self.max_frames, dtype=int)
             features = features[indices]
        
        # 4. Convert to tensor and normalize
        x = torch.tensor(features, dtype=torch.float32)
        x = (x - self.mean) / self.std
        
        if torch.isnan(x).any() or torch.isinf(x).any(): 
            raise ValueError("normalized features contain NaN or Inf values.")
            
        # Add batch dimension (1, T, D)
        return x.unsqueeze(0)
