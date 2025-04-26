import torch.nn as nn
import torch

class PronunciationClassifier(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        # Bidirectional LSTM with more layers and larger hidden dimension
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Fully connected layers with dropout
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM layer
        lstm_out, (hn, _) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers
        out = self.fc(attended)
        
        return self.sigmoid(out)
