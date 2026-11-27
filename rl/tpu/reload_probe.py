"""Check that the in-place vLLM weight reload is exact: reload the base text weights and require
bit-identical parameters and identical greedy output."""
import os
import sys
import time

sys.path.insert(0, "/rl")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from qrl.rollout import VLLMBackend, _vllm_param_snapshot  # noqa: E402

model = sys.argv[1]
be = VLLMBackend(model, 4608, 0)
from vllm import SamplingParams  # noqa: E402

prompt = be.tok.apply_chat_template([{"role": "user", "content": "What is 17*23? Answer briefly."}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
greedy = SamplingParams(temperature=0.0, max_tokens=40)
before_txt = be.llm.generate([prompt], greedy, use_tqdm=False)[0].outputs[0].text
before = be.llm.collective_rpc(_vllm_param_snapshot)[0]
sd = {}
for f in os.listdir(model):
    if f.endswith(".safetensors"):
        sd.update({k: v for k, v in load_file(os.path.join(model, f)).items() if k.startswith("model.language_model.")})
path = "/dev/shm/reload_probe.safetensors"
save_file(sd, path)
t = time.time()
be.load_weights(path)
print("reload_s", round(time.time() - t, 2))
after = be.llm.collective_rpc(_vllm_param_snapshot)[0]
bad = [n for n in before if n not in after or before[n].shape != after[n].shape or not torch.equal(before[n], after[n])]
print("params", len(before), "mismatched", len(bad), bad[:8])
for n in bad[:4]:
    print(n, tuple(before[n].shape), tuple(after[n].shape) if n in after else None)
after_txt = be.llm.generate([prompt], greedy, use_tqdm=False)[0].outputs[0].text
print("greedy_before", repr(before_txt[:80]))
print("greedy_after ", repr(after_txt[:80]))
# A perturbed reload must change the output, else the weights are not actually used.
sd2 = {k: (v * 0 if "mlp.down_proj" in k else v) for k, v in sd.items()}
save_file(sd2, path)
be.load_weights(path)
pert_txt = be.llm.generate([prompt], greedy, use_tqdm=False)[0].outputs[0].text
print("greedy_perturbed", repr(pert_txt[:80]))
print("RESULT", "OK" if not bad and before_txt == after_txt and pert_txt != before_txt else "FAIL")
