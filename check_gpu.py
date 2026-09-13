"""Step 1: verify the NVIDIA driver is visible to PyTorch and can run a CUDA kernel."""
import sys

import torch

print(f"python      : {sys.version.split()[0]}")
print(f"torch       : {torch.__version__}")
print(f"torch cuda  : {torch.version.cuda}")
print(f"cudnn       : {torch.backends.cudnn.version()}")
print(f"cuda avail  : {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("CUDA not available to torch")

dev = torch.cuda.current_device()
props = torch.cuda.get_device_properties(dev)
print(f"device      : {props.name}")
print(f"capability  : sm_{props.major}{props.minor}")
print(f"vram total  : {props.total_memory / 2**30:.1f} GiB")
free, total = torch.cuda.mem_get_info()
print(f"vram free   : {free / 2**30:.1f} GiB")
print(f"bf16 support: {torch.cuda.is_bf16_supported()}")

# Real kernel launch + numerical sanity check in fp32, fp16 and bf16.
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    a = torch.randn(2048, 2048, device="cuda", dtype=dtype)
    b = torch.randn(2048, 2048, device="cuda", dtype=dtype)
    c = (a @ b).float()
    ref = (a.float().cpu() @ b.float().cpu())
    err = (c.cpu() - ref).abs().max().item()
    print(f"matmul {str(dtype):15s} max abs err vs cpu fp32: {err:.4f}")
torch.cuda.synchronize()
print("OK: driver + CUDA kernels working")
