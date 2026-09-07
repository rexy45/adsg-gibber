import re
import torch
import numpy as np
from test_and_train import MicroTransformer

print("Reading existing model_weights.h...")

with open("model_weights.h", "r") as f:
    content = f.read()

# Match every static float array in the C header
pattern = re.compile(r'static const float ([a-zA-Z0-9_]+)\[(\d+)\]\s*=\s*\{([^}]+)\};', re.DOTALL)
matches = pattern.findall(content)

raw_arrays = {}
for c_name, count, body in matches:
    # Skip causal mask buffer if present
    if "causal_mask" in c_name:
        continue
    # Extract floating point numbers (including signs and scientific notation)
    tokens = [x.strip().rstrip('f') for x in body.split(',') if x.strip()]
    vals = []
    for t in tokens:
        try:
            vals.append(float(t))
        except ValueError:
            vals.append(0.0)
    raw_arrays[c_name] = np.array(vals, dtype=np.float32)

print(f"Extracted {len(raw_arrays)} weight tensors from C header.")

# Load into PyTorch model architecture
model = MicroTransformer()
sd = model.state_dict()

new_sd = {}
for k, v in sd.items():
    # Keep default causal_mask buffer untouched
    if "causal_mask" in k:
        new_sd[k] = v
        continue

    c_name = k.replace('.', '_')
    if c_name in raw_arrays:
        arr = raw_arrays[c_name]
        expected_elements = v.numel()
        if len(arr) == expected_elements:
            new_sd[k] = torch.tensor(arr.reshape(v.shape))
        else:
            print(f"Size mismatch for {k}: expected {expected_elements}, got {len(arr)}")
    else:
        print(f"Warning: {k} missing in header")

model.load_state_dict(new_sd)
torch.save(model.state_dict(), "adsg_model.pth")
print("\nSUCCESS: adsg_model.pth created in under 2 seconds! Zero re-training needed.")