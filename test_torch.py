import torch
import transformers

print("PyTorch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("CUDA device count:", torch.cuda.device_count())
print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        print(f"Device {i} capability: {cap}")
