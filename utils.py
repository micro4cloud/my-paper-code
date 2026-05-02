import numpy as np

def compute_rmse(pred, target):
    pred = np.array(pred)
    target = np.array(target)
    return np.sqrt(np.mean((pred - target) ** 2))

def compute_mae(pred, target):
    return np.mean(np.abs(np.array(pred) - np.array(target)))

def compute_r2(pred, target):
    target = np.array(target)
    pred = np.array(pred)
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    return 1 - (ss_res / ss_tot)