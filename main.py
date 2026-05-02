from train import train
from evaluate import evaluate

DATA_PATH = "data.npz"

if __name__ == "__main__":
    train(DATA_PATH)
    evaluate(DATA_PATH)