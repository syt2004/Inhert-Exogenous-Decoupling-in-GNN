import os
import numpy as np
from dataloader import DataLoader


def load_data(data_dir, batch_size,amount):
    data = {}
    for category in ["train", "val", "test"]:
        cat_data = np.load(os.path.join(data_dir, f"{category}_{amount}.npz"))
        xs, ys = cat_data["x"], cat_data["y"]
        data[category] = DataLoader(xs, ys, batch_size)
    return data
