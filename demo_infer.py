import torch
from test_and_train import Tokens, MicroTransformer, tokens_to_names, tokens_to_hex

print("=" * 70)
print("ADSG-GIBBER: AUTONOMOUS DISASTER SLM DEMO (136k PARAMS)")
print("Hardware Target: Seeed Studio XIAO ESP32-S3 (Xtensa LX7 @ 240 MHz)")
print("=" * 70)

# 1. Load trained brain instantly from checkpoint
model = MicroTransformer()
model.load_state_dict(torch.load("adsg_model.pth", weights_only=True))
model.eval()

# 2. Define a real crisis scenario:
# Zone 2, Earthquake, High Confidence, Route A BLOCKED, Route B OK
test_prompt = [
    Tokens.BOS,
    Tokens.ZONE_2,
    Tokens.SIG_QUAKE,
    Tokens.CONF_HIGH,
    Tokens.RTE_A_BLOCKED,
    Tokens.RTE_B_OK,
    Tokens.RES_EVAC_OK,
    Tokens.RES_AID_OK,
    Tokens.RES_ALARM_OK,
    Tokens.SEP
]

print(f"\n[Telemetry Inbound]")
print(f"Raw Hex : {tokens_to_hex(test_prompt)}")
print(f"Decoded : {tokens_to_names(test_prompt)}")

# 3. Model runs greedy autoregressive forward pass
input_tensor = torch.tensor([test_prompt], dtype=torch.long)
with torch.no_grad():
    output = model.generate(input_tensor, max_new_tokens=8)[0].tolist()

# 4. Extract generated decision sequence after SEP token
sep_idx = output.index(Tokens.SEP)
response = output[sep_idx + 1:]

print(f"\n[Autonomous Edge Decision]")
print(f"Raw Hex : {tokens_to_hex(response)}")
print(f"Decoded : {tokens_to_names(response)}")
print("=" * 70)