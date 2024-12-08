try:
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch Success!")
    print(f'Device: "{device}"')
except Exception as e:
    print(f"Error: {e}")
