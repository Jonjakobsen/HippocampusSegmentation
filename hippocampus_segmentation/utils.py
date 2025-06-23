import os
import torch

def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename)

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def create_dirs(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)
