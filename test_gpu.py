import torch

print("PyTorch version:", torch.__version__)

# Check for CUDA (NVIDIA GPUs)
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)
if cuda_available:
    print("GPU:", torch.cuda.get_device_name(0))

print("-" * 20)

# Check for MPS (Apple Silicon Macs)
mps_available = torch.backends.mps.is_available()
print("MPS available:", mps_available)
if mps_available:
    # Set the device to MPS
    device = torch.device("mps")
    print(f"Code is configured to use the {device.type.upper()} device.")

    # Create a tensor and explicitly place it on the MPS device
    x = torch.rand(5, 3).to(device)
    print("Tensor on device:", x.device)
    print("Tensor values:\n", x)
else:
    # If no MPS or CUDA is available, fall back to the CPU
    device = torch.device("cpu")
    print(f"Code is running on the {device.type.upper()} device.")