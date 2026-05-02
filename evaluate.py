import torch
from dataset import VLCDataset
from model import HybridModel
from utils import compute_rmse

def evaluate(data_path):
    dataset = VLCDataset(data_path)
    model = HybridModel(input_dim=dataset.X.shape[1])
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    preds, targets = [], []

    for x, y in dataset:
        with torch.no_grad():
            pred = model(x.unsqueeze(0))
        preds.append(pred.item())
        targets.append(y.item())

    rmse = compute_rmse(preds, targets)
    print("RMSE:", rmse)