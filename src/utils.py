import torch

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_device(tensor_or_model, device):
    return tensor_or_model.to(device)