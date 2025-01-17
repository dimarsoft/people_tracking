import torch


def show_torch():
    """
    Версии PyTorch
    Returns:

    """
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDNN version: {torch.backends.cudnn.version()}')
    print(f'Available GPU devices: {torch.cuda.device_count()}')
    print(f'Device Name: {torch.cuda.get_device_name()}')

show_torch()

