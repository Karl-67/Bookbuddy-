import torch.nn as nn
import torch
import torchaudio.pipelines as pipelines

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


class TransferLearningPronunciationClassifier(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.3, freeze_extractor=True):
        super().__init__()
        
        # Load pre-trained wav2vec2 model
        bundle = pipelines.WAV2VEC2_BASE
        self.feature_extractor = bundle.get_model()
        
        # Freeze the feature extractor
        if freeze_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        # Feature dimension from wav2vec2 base model
        feature_dim = 768
        
        # Bidirectional LSTM to process the extracted features
        self.lstm = nn.LSTM(
            feature_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Fully connected layers
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
        # x shape: [batch_size, time]
        # Extract features using wav2vec2
        with torch.no_grad() if self.feature_extractor.training == False else torch.enable_grad():
            features, _ = self.feature_extractor.extract_features(x)
        
        # Use the last layer features
        x = features[-1]
        
        # Process with LSTM
        lstm_out, (hn, _) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers
        out = self.fc(attended)
        
        return self.sigmoid(out)


class PhonemeEmbeddingPronunciationClassifier(nn.Module):
    def __init__(self, mfcc_dim=13, phoneme_vocab_size=50, phoneme_embed_dim=64, 
                 hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        # Phoneme embedding layer
        self.phoneme_embedding = nn.Embedding(phoneme_vocab_size, phoneme_embed_dim)
        
        # Bidirectional LSTM for audio features
        self.audio_lstm = nn.LSTM(
            mfcc_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Bidirectional LSTM for phoneme embeddings
        self.phoneme_lstm = nn.LSTM(
            phoneme_embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism for audio
        self.audio_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Attention mechanism for phonemes
        self.phoneme_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Fully connected layers with dropout
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # Combine audio and phoneme features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, audio_features, phoneme_indices):
        # Process audio features
        audio_lstm_out, _ = self.audio_lstm(audio_features)
        
        # Apply attention to audio features
        audio_attn_weights = self.audio_attention(audio_lstm_out)
        audio_attn_weights = torch.softmax(audio_attn_weights, dim=1)
        audio_attended = torch.sum(audio_attn_weights * audio_lstm_out, dim=1)
        
        # Process phoneme embeddings
        phoneme_embeddings = self.phoneme_embedding(phoneme_indices)
        phoneme_lstm_out, _ = self.phoneme_lstm(phoneme_embeddings)
        
        # Apply attention to phoneme features
        phoneme_attn_weights = self.phoneme_attention(phoneme_lstm_out)
        phoneme_attn_weights = torch.softmax(phoneme_attn_weights, dim=1)
        phoneme_attended = torch.sum(phoneme_attn_weights * phoneme_lstm_out, dim=1)
        
        # Combine features
        combined = torch.cat([audio_attended, phoneme_attended], dim=1)
        
        # Fully connected layers
        out = self.fc(combined)
        
        return self.sigmoid(out)


class RegressionPronunciationClassifier(nn.Module):
    """
    Regression-based model that directly outputs a pronunciation score without sigmoid.
    """
    def __init__(self, input_dim=13, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        # Bidirectional LSTM
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
        
        # Fully connected layers with dropout - no sigmoid activation at the end
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # LSTM layer
        lstm_out, (hn, _) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers - outputs raw score (0 to 1 range, no sigmoid)
        out = self.fc(attended)
        
        return out


class ReferenceBasedPronunciationModel(nn.Module):
    """
    Model that directly compares user audio with reference audio
    to evaluate pronunciation quality.
    """
    def __init__(self, input_dim=13, hidden_dim=128):
        super().__init__()
        
        # Siamese network architecture
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Comparison network
        self.comparison = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, user_audio, reference_audio):
        # Extract features from both inputs
        user_features = self.feature_extractor(user_audio)
        ref_features = self.feature_extractor(reference_audio)
        
        # Global average pooling over time dimension
        user_features = torch.mean(user_features, dim=1)
        ref_features = torch.mean(ref_features, dim=1)
        
        # Concatenate features
        combined = torch.cat([user_features, ref_features], dim=1)
        
        # Calculate similarity score
        similarity = self.comparison(combined)
        
        return similarity
