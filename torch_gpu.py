"""
Вывод версии torch.
"""

import os
import platform
import subprocess

import torch

def get_nvcc_version() -> str:
    """
    Получиться версию
    Returns:

    """
    command = "nvcc  --version"

    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True)

        output = result.stdout.strip()
        return output
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr}")

    return ""

def print_nvcc_version() -> None:
    """
    Вывод информации о CUDA
    Returns:

    """

    print('*'*10)

    print('nvcc info:')
    print(get_nvcc_version())
    print('*'*10)

def print_os_info() -> None:
    """
    Вывод информации об ОС
    Returns:

    """
    # OS Name
    os_name = os.name  # 'posix', 'nt', 'os2', etc.

    # Detailed OS Information
    system_name = platform.system()  # E.g., 'Linux', 'Windows', 'Darwin'
    release = platform.release()  # OS release, e.g., '10' or '5.15.0-73-generic'
    version = platform.version()  # OS version
    architecture = platform.architecture()[0]  # E.g., '64bit' or '32bit'
    machine = platform.machine()  # E.g., 'x86_64' or 'AMD64'

    # Print the information
    print('*'*10)
    print(f"OS Name (os.name): {os_name}")
    print(f"System Name (platform.system()): {system_name}")
    print(f"Release: {release}")
    print(f"Version: {version}")
    print(f"Architecture: {architecture}")
    print(f"Machine: {machine}")

def show_torch() -> None:
    """
    Версии PyTorch
    Returns:

    """
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA version: {torch.version.cuda}')
    print_nvcc_version()
    print(f'CUDNN version: {torch.backends.cudnn.version()}')
    print(f'Available GPU devices: {torch.cuda.device_count()}')
    print(f'Device Name: {torch.cuda.get_device_name()}')


if __name__ == '__main__':
    show_torch()
    print_os_info()
