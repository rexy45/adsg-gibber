#!/usr/bin/env python3
"""
ADSG-Gibber: Instant Telemetry Stream Tester
Paste a raw hex stream directly from INPUT_TEMPLATES.txt and hit Enter.
"""

import torch
from test_and_train import Tokens, MicroTransformer, tokens_to_names, tokens_to_hex

def load_engine():
    model = MicroTransformer()
    model.load_state_dict(torch.load("adsg_model.pth", weights_only=True))
    model.eval()
    return model

def run_inference(model, prompt):
    print("\n" + "=" * 70)
    print("INCOMING SENSOR TELEMETRY (1-BYTE PROTOCOL)")
    print(f"Raw Hex Stream : {tokens_to_hex(prompt)}")
    print(f"Decoded Packet : {tokens_to_names(prompt)}")
    print("-" * 70)

    input_tensor = torch.tensor([prompt], dtype=torch.long)
    with torch.no_grad():
        output = model.generate(input_tensor, max_new_tokens=8)[0].tolist()

    sep_idx = output.index(Tokens.SEP)
    decision = output[sep_idx + 1:]

    print("AUTONOMOUS EDGE DECISION (INFERENCE LATENCY: ~12 ms)")
    print(f"Raw Hex Stream : {tokens_to_hex(decision)}")
    print(f"Decoded Action : {tokens_to_names(decision)}")
    print("=" * 70 + "\n")

def main():
    print("=" * 70)
    print("ADSG-GIBBER EDGE INFERENCE ENGINE (136k PARAMS)")
    print("Target: Seeed Studio XIAO ESP32-S3 (Xtensa LX7 @ 240 MHz)")
    print("=" * 70)
    
    model = load_engine()

    while True:
        print("\nOpen 'INPUT_TEMPLATES.txt', copy any RAW HEX STREAM line, and paste it here.")
        raw = input("Paste Hex Stream (or 'q' to quit) > ").strip()
        
        if raw.lower() in ('q', 'exit'):
            break
        if not raw:
            continue

        try:
            prompt = [int(tok, 16) for tok in raw.replace(',', ' ').split()]
            if prompt[0] != Tokens.BOS: prompt.insert(0, Tokens.BOS)
            if prompt[-1] != Tokens.SEP: prompt.append(Tokens.SEP)
            run_inference(model, prompt)
        except Exception as e:
            print(f"[!] Invalid hex input format: {e}. Example: 0x01 0x12 0x23 0x62 0x41 0x42 0x44 0x46 0x48 0x03")

if __name__ == "__main__":
    main()