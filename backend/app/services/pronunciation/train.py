import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from services.pronunciation.dataset import PronunciationDataset
from services.pronunciation.model import PronunciationClassifier

# Load dataset
dataset = PronunciationDataset("data/processed", "data/labels.csv")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model
model = PronunciationClassifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train loop
for epoch in range(20):
    total_loss = 0
    for x, y in loader:
        output = model(x).squeeze()
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# Save model
torch.save(model.state_dict(), "models/pronunciation_model.pt")
