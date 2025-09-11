import torch
ckpt = torch.load("liquid/best.pt", map_location="cpu")

ckpt['model'].names = {0: 'Gel', 1: 'Stable', 2: 'Air'}
torch.save(ckpt, "liquid/best_renamed.pt")
