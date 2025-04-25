import os
import sys
import numpy as np

# Add the app directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(app_dir)

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from services.pronunciation.dataset import PronunciationDataset
from services.pronunciation.model import PronunciationClassifier

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load dataset with correct paths
reference_dir = "C:/Users/yurig/OneDrive/Desktop/reference_renamed"
train_dir = "C:/Users/yurig/OneDrive/Desktop/train_renamed"
dataset = PronunciationDataset(reference_dir, train_dir)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model with weight decay for regularization
model = PronunciationClassifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)  # Lower learning rate, added weight decay

# Training parameters
num_epochs = 3  # 3 epochs as requested
checkpoint_frequency = 1  # Save checkpoint every epoch
best_loss = float('inf')

# Train loop
for epoch in range(num_epochs):
    total_loss = 0
    all_outputs = []
    model.train()  # Set model to training mode
    
    for x, y in loader:
        x = x.transpose(1, 2)  # Shape: [B, 13, 500] → [B, 500, 13]
        output = model(x).squeeze()
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Collect predictions for analysis
        all_outputs.extend(output.detach().cpu().numpy())
    
    # Calculate statistics
    outputs_array = np.array(all_outputs)
    avg_loss = total_loss / len(loader)
    
    # Print detailed output distribution
    print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")
    print(f"Prediction stats - Min: {outputs_array.min():.4f}, Max: {outputs_array.max():.4f}, Mean: {outputs_array.mean():.4f}")
    print(f"Distribution - <0.2: {np.mean(outputs_array < 0.2):.2f}, 0.2-0.8: {np.mean((outputs_array >= 0.2) & (outputs_array <= 0.8)):.2f}, >0.8: {np.mean(outputs_array > 0.8):.2f}")
    
    # Save checkpoint every checkpoint_frequency epochs
    if (epoch + 1) % checkpoint_frequency == 0:
        checkpoint_path = f"models/pronunciation_model_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, "models/pronunciation_model_best.pt")
        print(f"New best model saved with loss: {best_loss:.4f}")

# Save final model
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
}, "models/pronunciation_model_final.pt")
print("Training completed. Final model saved.")
