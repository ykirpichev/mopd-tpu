"""Inside the vllm-torchtpu container: versions, one-chip pinning, and a bf16 matmul + backward on "tpu"."""
import os
import time

import torch

t0 = time.time()
import torch_tpu  # noqa: E402,F401
import transformers  # noqa: E402
import vllm  # noqa: E402

print("versions", {"torch": torch.__version__, "torch_tpu": getattr(torch_tpu, "__version__", "?"),
                   "transformers": transformers.__version__, "vllm": vllm.__version__})
print("env", {k: v for k, v in os.environ.items() if k.startswith(("TPU_", "TORCH_TPU"))})
dev = torch.device("tpu")
try:
    print("device_count", torch.tpu.device_count())
except Exception as e:
    print("device_count n/a", repr(e)[:200])
x = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16, requires_grad=True)
y = (x @ x).float().sum()
y.backward()
print("matmul+backward ok", float(y), x.grad.shape, round(time.time() - t0, 1), "s")
t1 = time.time()
for _ in range(20):
    z = x @ x
float(z.float().sum())
dt = (time.time() - t1) / 20
print("bf16 4096^3 matmul", round(dt * 1e3, 2), "ms ->", round(2 * 4096**3 / dt / 1e12, 1), "TFLOP/s")
from transformers import Qwen3_5ForCausalLM  # noqa: E402,F401

print("Qwen3_5ForCausalLM import ok")
