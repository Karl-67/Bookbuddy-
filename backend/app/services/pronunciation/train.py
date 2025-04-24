import os
import sys
import argparse

# Add the app directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(app_dir)

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from services.pronunciation.dataset import (
    PronunciationDataset, 
    Wav2Vec2PronunciationDataset,
    PhonemeDataset,
    ReferencePronunciationDataset
)
from services.pronunciation.model import (
    PronunciationClassifier, 
    TransferLearningPronunciationClassifier,
    PhonemeEmbeddingPronunciationClassifier,
    RegressionPronunciationClassifier,
    ReferenceBasedPronunciationModel
)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train pronunciation model')
parser.add_argument('--model_type', type=str, default='mfcc', 
                    choices=['mfcc', 'wav2vec2', 'phoneme', 'regression', 'reference'],
                    help='Model type to train')
parser.add_argument('--data_path', type=str, required=True,
                    help='Path to dataset')
parser.add_argument('--reference_path', type=str, default=None,
                    help='Path to reference audio dataset (for reference model)')
parser.add_argument('--cmu_dict_path', type=str, default=None,
                    help='Path to CMU dictionary (for phoneme model)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training')
parser.add_argument('--epochs', type=int, default=40,
                    help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate')
args = parser.parse_args()

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Configuration
config = {
    "model_type": args.model_type,  # mfcc, wav2vec2, phoneme, regression, reference
    "data_path": args.data_path,
    "reference_path": args.reference_path,
    "cmu_dict_path": args.cmu_dict_path,
    "batch_size": args.batch_size,
    "num_epochs": args.epochs,
    "learning_rate": args.lr,
    "checkpoint_frequency": 5,
    "model_dir": "models",
}

print(f"Training with configuration: {config}")

# Select dataset and model based on configuration
if config["model_type"] == "wav2vec2":
    print("Using Wav2Vec2 transfer learning model...")
    dataset = Wav2Vec2PronunciationDataset(config["data_path"])
    model = TransferLearningPronunciationClassifier(freeze_extractor=True)
    criterion = nn.BCELoss()
elif config["model_type"] == "phoneme":
    print("Using Phoneme Embedding model...")
    dataset = PhonemeDataset(config["data_path"], config["cmu_dict_path"])
    model = PhonemeEmbeddingPronunciationClassifier(
        phoneme_vocab_size=50,  # Fixed vocab size for phonemes
        phoneme_embed_dim=64
    )
    criterion = nn.BCELoss()
elif config["model_type"] == "regression":
    print("Using Regression model...")
    dataset = PronunciationDataset(config["data_path"])
    model = RegressionPronunciationClassifier()
    criterion = nn.MSELoss()  # Using MSE loss for regression
elif config["model_type"] == "reference":
    print("Using Reference-based model...")
    if config["reference_path"] is None:
        raise ValueError("Reference path must be specified for reference model")
    dataset = ReferencePronunciationDataset(config["data_path"], config["reference_path"])
    model = ReferenceBasedPronunciationModel()
    criterion = nn.MSELoss()  # Using MSE loss for similarity learning
else:  # Default to MFCC model
    print("Using original MFCC-based model...")
    dataset = PronunciationDataset(config["data_path"])
    model = PronunciationClassifier()
    criterion = nn.BCELoss()

# Create data loader
loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# Training parameters
best_loss = float('inf')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Train loop
for epoch in range(config["num_epochs"]):
    total_loss = 0
    model.train()  # Set model to training mode
    
    for batch_idx, batch_data in enumerate(loader):
        # Handle different input formats based on model type
        if config["model_type"] == "phoneme":
            # PhonemeDataset returns (mfcc, phoneme_indices, score)
            mfcc, phoneme_indices, y = batch_data
            mfcc = mfcc.to(device)
            phoneme_indices = phoneme_indices.to(device)
            y = y.to(device)
            
            # Transpose MFCC features
            mfcc = mfcc.transpose(1, 2)  # Shape: [B, 13, 500] → [B, 500, 13]
            
            # Forward pass with both mfcc and phoneme inputs
            output = model(mfcc, phoneme_indices).squeeze()
        elif config["model_type"] == "wav2vec2":
            # Wav2Vec2Dataset returns (waveform, score)
            x, y = batch_data
            x = x.to(device)
            y = y.to(device)
            
            # No need to transform input for wav2vec2
            output = model(x).squeeze()
        elif config["model_type"] == "reference":
            # ReferencePronunciationDataset returns (user_mfcc, ref_mfcc, similarity_score)
            user_mfcc, ref_mfcc, y = batch_data
            user_mfcc = user_mfcc.to(device)
            ref_mfcc = ref_mfcc.to(device)
            y = y.to(device)
            
            # Transpose MFCC features
            user_mfcc = user_mfcc.transpose(1, 2)  # Shape: [B, 13, 500] → [B, 500, 13]
            ref_mfcc = ref_mfcc.transpose(1, 2)   # Shape: [B, 13, 500] → [B, 500, 13]
            
            # Forward pass with both user and reference inputs
            output = model(user_mfcc, ref_mfcc).squeeze()
        else:
            # Regular datasets return (mfcc/features, score)
            x, y = batch_data
            x = x.to(device)
            y = y.to(device)
            
            # Transpose MFCC features
            x = x.transpose(1, 2)  # Shape: [B, 13, 500] → [B, 500, 13]
            
            output = model(x).squeeze()
        
        # Calculate loss
        loss = criterion(output, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Print batch progress
        if batch_idx % 5 == 0:
            print(f"Epoch {epoch+1}/{config['num_epochs']} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{config['num_epochs']} | Avg Loss: {avg_loss:.4f}")
    
    # Save checkpoint every checkpoint_frequency epochs
    if (epoch + 1) % config["checkpoint_frequency"] == 0:
        checkpoint_path = f"{config['model_dir']}/pronunciation_{config['model_type']}_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': config,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_path = f"{config['model_dir']}/pronunciation_{config['model_type']}_best.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
            'config': config,
        }, best_model_path)
        print(f"New best model saved with loss: {best_loss:.4f}")

# Save final model
torch.save({
    'epoch': config["num_epochs"],
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
    'config': config,
}, f"{config['model_dir']}/pronunciation_{config['model_type']}_final.pt")
print("Training completed. Final model saved.")
