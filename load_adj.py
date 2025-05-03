import pickle

def load_adj(pkl_path):
    with open(pkl_path, 'rb') as f:
        adj_mx = pickle.load(f)
    return adj_mx