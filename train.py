import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import VLCDataset
from model import HybridModel
from utils import compute_rmse

def train(data_path, epochs=50):
    dataset = VLCDataset(data_path)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = HybridModel(input_dim=dataset.X.shape[1])
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "model.pth")
    return model