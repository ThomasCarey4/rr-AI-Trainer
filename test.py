import torch
import time

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0))

# Create a large tensor operation to test GPU performance
def test_performance(device_name):
    # Create large tensors
    size = 10000
    a = torch.randn(size, size, device=device_name)
    b = torch.randn(size, size, device=device_name)
    
    # Time matrix multiplication
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize() if device_name == 'cuda' else None
    end = time.time()
    
    return end - start

# Test on CPU
print("\nTesting on CPU...")
cpu_time = test_performance('cpu')
print(f"CPU time: {cpu_time:.4f} seconds")

# Test on GPU
try:
    print("\nTesting on GPU...")
    gpu_time = test_performance('cuda')
    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"GPU speedup: {cpu_time/gpu_time:.2f}x faster than CPU")
except Exception as e:
    print("Error during GPU test:", e)