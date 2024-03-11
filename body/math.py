import numpy as np
import torch
def rotate_x(a, device="cuda"):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1,  0, 0, 0], 
                     [0,  c, s, 0], 
                     [0, -s, c, 0], 
                     [0,  0, 0, 1]]).to(torch.float32).to(device)

def rotate_y(a, device="cuda"):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[ c, 0, s, 0], 
                     [ 0, 1, 0, 0], 
                     [-s, 0, c, 0], 
                     [ 0, 0, 0, 1]]).to(torch.float32).to(device)

def rotate_z(a, device="cuda"): # chat
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[c, -s, 0, 0], 
                     [s,  c, 0, 0], 
                     [0,  0, 1, 0], 
                     [0,  0, 0, 1]]).to(torch.float32).to(device)