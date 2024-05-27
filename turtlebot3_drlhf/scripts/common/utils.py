import torch

def check_gpu():
    print("gpu torch available: ", torch.cuda.is_available())
    if (torch.cuda.is_available()):
        print("device name: ", torch.cuda.get_device_name(0))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
