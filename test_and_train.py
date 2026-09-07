#!/usr/bin/env python3
"""
ADSG-Gibber Micro-Transformer for Seeed Studio XIAO ESP32-S3
Self-contained training, evaluation, checkpointing, and C-header export script.
"""

import os
import sys
import math
import struct
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional

# ============================================================================
# ADSG-Gibber Token Protocol (0x00 - 0xFF)
# ============================================================================
class Tokens:
    PAD = 0x00
    BOS = 0x01
    EOS = 0x02
    SEP = 0x03

    ZONE_1 = 0x11
    ZONE_2 = 0x12
    ZONE_3 = 0x13
    ZONE_4 = 0x14

    SIG_FLOOD = 0x20
    SIG_RAIN_HIGH = 0x21
    SIG_DRAIN_OVF = 0x22
    SIG_QUAKE = 0x23
    SIG_ELEC_FIRE = 0x24
    SIG_LPG_LEAK = 0x25
    SIG_WATER_LEAK = 0x26
    SIG_INTRUSION = 0x27
    SIG_MED_REQ = 0x28

    RTE_A_OK = 0x40
    RTE_A_BLOCKED = 0x41
    RTE_B_OK = 0x42
    RTE_B_BLOCKED = 0x43
    RES_EVAC_OK = 0x44
    RES_EVAC_NO = 0x45
    RES_AID_OK = 0x46
    RES_AID_NO = 0x47
    RES_ALARM_OK = 0x48
    RES_ALARM_NO = 0x49

    CONF_LOW = 0x60
    CONF_MED = 0x61
    CONF_HIGH = 0x62
    CONF_CRIT = 0x63

    CMD_EVACUATE = 0x80
    CMD_MONITOR = 0x81
    CMD_ALERT = 0x82
    CMD_DISP_AID = 0x83
    CMD_ROUTE_A = 0x84
    CMD_ROUTE_B = 0x85
    CMD_ALARM_ON = 0x86
    CMD_ALARM_OFF = 0x87
    CMD_NOTIFY = 0x88

    PRIO_MONITOR = 0xB0
    PRIO_WARNING = 0xB1
    PRIO_HIGH = 0xB2
    PRIO_CRITICAL = 0xB3

VOCAB_SIZE = 256
MAX_SEQ_LEN = 32
D_MODEL = 64
D_FF = 128
N_LAYERS = 3
N_HEADS = 4
HEAD_DIM = 16
PAD_IDX = Tokens.PAD

# Token name mapping for pretty printing
TOKEN_NAMES = {
    Tokens.PAD: "PAD", Tokens.BOS: "BOS", Tokens.EOS: "EOS", Tokens.SEP: "SEP",
    Tokens.ZONE_1: "ZONE_1", Tokens.ZONE_2: "ZONE_2", Tokens.ZONE_3: "ZONE_3", Tokens.ZONE_4: "ZONE_4",
    Tokens.SIG_FLOOD: "SIG_FLOOD", Tokens.SIG_RAIN_HIGH: "SIG_RAIN_HIGH", Tokens.SIG_DRAIN_OVF: "SIG_DRAIN_OVF",
    Tokens.SIG_QUAKE: "SIG_QUAKE", Tokens.SIG_ELEC_FIRE: "SIG_ELEC_FIRE", Tokens.SIG_LPG_LEAK: "SIG_LPG_LEAK",
    Tokens.SIG_WATER_LEAK: "SIG_WATER_LEAK", Tokens.SIG_INTRUSION: "SIG_INTRUSION", Tokens.SIG_MED_REQ: "SIG_MED_REQ",
    Tokens.RTE_A_OK: "RTE_A_OK", Tokens.RTE_A_BLOCKED: "RTE_A_BLOCKED", Tokens.RTE_B_OK: "RTE_B_OK",
    Tokens.RTE_B_BLOCKED: "RTE_B_BLOCKED", Tokens.RES_EVAC_OK: "RES_EVAC_OK", Tokens.RES_EVAC_NO: "RES_EVAC_NO",
    Tokens.RES_AID_OK: "RES_AID_OK", Tokens.RES_AID_NO: "RES_AID_NO",
    Tokens.RES_ALARM_OK: "RES_ALARM_OK", Tokens.RES_ALARM_NO: "RES_ALARM_NO",
    Tokens.CONF_LOW: "CONF_LOW", Tokens.CONF_MED: "CONF_MED", Tokens.CONF_HIGH: "CONF_HIGH", Tokens.CONF_CRIT: "CONF_CRIT",
    Tokens.CMD_EVACUATE: "CMD_EVACUATE", Tokens.CMD_MONITOR: "CMD_MONITOR", Tokens.CMD_ALERT: "CMD_ALERT",
    Tokens.CMD_DISP_AID: "CMD_DISP_AID", Tokens.CMD_ROUTE_A: "CMD_ROUTE_A", Tokens.CMD_ROUTE_B: "CMD_ROUTE_B",
    Tokens.CMD_ALARM_ON: "CMD_ALARM_ON", Tokens.CMD_ALARM_OFF: "CMD_ALARM_OFF", Tokens.CMD_NOTIFY: "CMD_NOTIFY",
    Tokens.PRIO_MONITOR: "PRIO_MONITOR", Tokens.PRIO_WARNING: "PRIO_WARNING", Tokens.PRIO_HIGH: "PRIO_HIGH", Tokens.PRIO_CRITICAL: "PRIO_CRITICAL",
}

def token_name(t: int) -> str:
    return TOKEN_NAMES.get(t, f"UNK(0x{t:02X})")

def tokens_to_hex(tokens: List[int]) -> str:
    return " ".join(f"0x{t:02X}" for t in tokens)

def tokens_to_names(tokens: List[int]) -> str:
    return " ".join(token_name(t) for t in tokens)

# ============================================================================
# Model Components
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight

def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.head_dim = HEAD_DIM
        self.d_model = D_MODEL
        self.scale = HEAD_DIM ** -0.5

        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.proj = nn.Linear(D_MODEL, D_MODEL, bias=False)

        self.register_buffer("causal_mask", torch.triu(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN) * float('-inf'), diagonal=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att + self.causal_mask[:T, :T]
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, D_FF, bias=False)
        self.fc2 = nn.Linear(D_FF, D_MODEL, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(fast_gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = RMSNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2 = RMSNorm(D_MODEL)
        self.ffn = FeedForward()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class MicroTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Parameter(torch.zeros(1, MAX_SEQ_LEN, D_MODEL))
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.ln_f = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1), ignore_index=PAD_IDX)
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 16, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = prompt[:, -MAX_SEQ_LEN:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            prompt = torch.cat([prompt, next_token], dim=1)
            if next_token.item() == Tokens.EOS:
                break
        return prompt

# ============================================================================
# Part A: Synthetic Dataset Generator
# ============================================================================
class ADSGDataset(Dataset):
    def __init__(self, num_samples: int = 80000):
        self.num_samples = num_samples
        self.samples = self._generate_samples()

    def _pad(self, seq: List[int]) -> List[int]:
        return seq + [Tokens.PAD] * (MAX_SEQ_LEN - len(seq))

    def _make_sample(self, zone: int, hazard: int, conf: int, route_a: int, route_b: int, res_evac: int, res_aid: int, res_alarm: int,
                     priority: int, action: int, route_cmd: int, alarm_cmd: int, notify_cmd: int = Tokens.PAD) -> List[int]:
        prompt = [Tokens.BOS, zone, hazard, conf, route_a, route_b, res_evac, res_aid, res_alarm, Tokens.SEP]
        target = [priority, action, route_cmd, alarm_cmd]
        if notify_cmd != Tokens.PAD:
            target.append(notify_cmd)
        target.append(Tokens.EOS)
        full = prompt + target
        return self._pad(full)

    def _generate_samples(self) -> List[List[int]]:
        samples = []
        scenarios_per_type = self.num_samples // 10

        for _ in range(scenarios_per_type):
            # Scenario 1: Single confirmed flood -> Evacuate open route + Alarm
            zone = random.choice([Tokens.ZONE_1, Tokens.ZONE_2, Tokens.ZONE_3, Tokens.ZONE_4])
            route_a = random.choice([Tokens.RTE_A_OK, Tokens.RTE_A_BLOCKED])
            route_b = Tokens.RTE_B_OK
            cmd_route = Tokens.CMD_ROUTE_B if route_a == Tokens.RTE_A_BLOCKED else Tokens.CMD_ROUTE_A
            samples.append(self._make_sample(
                zone, Tokens.SIG_FLOOD, Tokens.CONF_HIGH,
                route_a, route_b, Tokens.RES_EVAC_OK, Tokens.RES_AID_NO, Tokens.RES_ALARM_OK,
                Tokens.PRIO_HIGH, Tokens.CMD_EVACUATE, cmd_route, Tokens.CMD_ALARM_ON
            ))

            # Scenario 2: Multi-sensor rain/drain confirmation -> Escalate to Critical
            zone = random.choice([Tokens.ZONE_1, Tokens.ZONE_2, Tokens.ZONE_3, Tokens.ZONE_4])
            samples.append(self._make_sample(
                zone, Tokens.SIG_RAIN_HIGH, Tokens.CONF_CRIT,
                Tokens.RTE_A_OK, Tokens.RTE_B_OK, Tokens.RES_EVAC_OK, Tokens.RES_AID_OK, Tokens.RES_ALARM_OK,
                Tokens.PRIO_CRITICAL, Tokens.CMD_EVACUATE, Tokens.CMD_ROUTE_A, Tokens.CMD_ALARM_ON
            ))

            # Scenario 3: Conflicting / weak sensor data -> Monitor / Warning state
            zone = random.choice([Tokens.ZONE_1, Tokens.ZONE_2, Tokens.ZONE_3, Tokens.ZONE_4])
            hazard = random.choice([Tokens.SIG_RAIN_HIGH, Tokens.SIG_WATER_LEAK])
            conf = random.choice([Tokens.CONF_LOW, Tokens.CONF_MED])
            samples.append(self._make_sample(
                zone, hazard, conf,
                Tokens.RTE_A_OK, Tokens.RTE_B_OK, Tokens.RES_EVAC_NO, Tokens.RES_AID_NO, Tokens.RES_ALARM_NO,
                Tokens.PRIO_WARNING, Tokens.CMD_MONITOR, Tokens.CMD_ROUTE_A, Tokens.CMD_ALARM_OFF
            ))

            # Scenario 4: Earthquake + Route A blocked -> Dynamic rerouting to Route B + Standby Aid
            zone = random.choice([Tokens.ZONE_1, Tokens.ZONE_2, Tokens.ZONE_3, Tokens.ZONE_4])
            samples.append(self._make_sample(
                zone, Tokens.SIG_QUAKE, Tokens.CONF_HIGH,
                Tokens.RTE_A_BLOCKED, Tokens.RTE_B_OK, Tokens.RES_EVAC_OK, Tokens.RES_AID_OK, Tokens.RES_ALARM_OK,
                Tokens.PRIO_HIGH, Tokens.CMD_EVACUATE, Tokens.CMD_ROUTE_B, Tokens.CMD_ALARM_ON
            ))

            # Scenario 5: Indoor electrical fire -> Immediate Critical Evacuation + Alarm
            zone = random.choice([Tokens.ZONE_1, Tokens.ZONE_2, Tokens.ZONE_3, Tokens.ZONE_4])
            samples.append(self._make_sample(
                zone, Tokens.SIG_ELEC_FIRE, Tokens.CONF_CRIT,
                Tokens.RTE_A_OK, Tokens.RTE_B_OK, Tokens.RES_EVAC_OK, Tokens.RES_AID_OK, Tokens.RES_ALARM_OK,
                Tokens.PRIO_CRITICAL, Tokens.CMD_EVACUATE, Tokens.CMD_ROUTE_A, Tokens.CMD_ALARM_ON
            ))

            # Scenario 6: LPG gas leak -> High Alert + Alarm (Forbid water/sprinkler actions)
            zone = random.choice([Tokens.ZONE_1, Tokens.ZONE_2, Tokens.ZONE_3, Tokens.ZONE_4])
            samples.append(self._make_sample(
                zone, Tokens.SIG_LPG_LEAK, Tokens.CONF_HIGH,
                Tokens.RTE_A_OK, Tokens.RTE_B_OK, Tokens.RES_EVAC_OK, Tokens.RES_AID_NO, Tokens.RES_ALARM_OK,
                Tokens.PRIO_HIGH, Tokens.CMD_ALERT, Tokens.CMD_ROUTE_A, Tokens.CMD_ALARM_ON
            ))

            # Scenario 7: Escalating water leak -> Monitor first, escalate upon drain overflow
            zone = random.choice([Tokens.ZONE_1, Tokens.ZONE_2, Tokens.ZONE_3, Tokens.ZONE_4])
            escalate = random.random() < 0.5
            hazard = Tokens.SIG_DRAIN_OVF if escalate else Tokens.SIG_WATER_LEAK
            conf = Tokens.CONF_HIGH if escalate else Tokens.CONF_MED
            priority = Tokens.PRIO_HIGH if escalate else Tokens.PRIO_WARNING
            action = Tokens.CMD_EVACUATE if escalate else Tokens.CMD_MONITOR
            alarm = Tokens.CMD_ALARM_ON if escalate else Tokens.CMD_ALARM_OFF
            samples.append(self._make_sample(
                zone, hazard, conf,
                Tokens.RTE_A_OK, Tokens.RTE_B_OK, Tokens.RES_EVAC_OK, Tokens.RES_AID_NO, Tokens.RES_ALARM_OK,
                priority, action, Tokens.CMD_ROUTE_A, alarm
            ))

            # Scenario 8: Multi-zone concurrent disasters -> Priority hierarchy (Fire > Quake > Flood)
            zones = random.sample([Tokens.ZONE_1, Tokens.ZONE_2, Tokens.ZONE_3, Tokens.ZONE_4], 2)
            hazard1 = random.choice([Tokens.SIG_ELEC_FIRE, Tokens.SIG_QUAKE, Tokens.SIG_FLOOD])
            hazard2 = random.choice([Tokens.SIG_ELEC_FIRE, Tokens.SIG_QUAKE, Tokens.SIG_FLOOD])
            
            def hazard_priority(h):
                if h == Tokens.SIG_ELEC_FIRE: return 3
                if h == Tokens.SIG_QUAKE: return 2
                return 1

            primary_hazard = hazard1 if hazard_priority(hazard1) >= hazard_priority(hazard2) else hazard2
            primary_zone = zones[0] if primary_hazard == hazard1 else zones[1]
            conf = Tokens.CONF_CRIT if primary_hazard == Tokens.SIG_ELEC_FIRE else Tokens.CONF_HIGH
            priority = Tokens.PRIO_CRITICAL if primary_hazard == Tokens.SIG_ELEC_FIRE else Tokens.PRIO_HIGH
            samples.append(self._make_sample(
                primary_zone, primary_hazard, conf,
                Tokens.RTE_A_OK, Tokens.RTE_B_OK, Tokens.RES_EVAC_OK, Tokens.RES_AID_OK, Tokens.RES_ALARM_OK,
                priority, Tokens.CMD_EVACUATE, Tokens.CMD_ROUTE_A, Tokens.CMD_ALARM_ON
            ))

            # Scenario 9: Medical request -> Dispatch aid + Notify
            zone = random.choice([Tokens.ZONE_1, Tokens.ZONE_2, Tokens.ZONE_3, Tokens.ZONE_4])
            samples.append(self._make_sample(
                zone, Tokens.SIG_MED_REQ, Tokens.CONF_HIGH,
                Tokens.RTE_A_OK, Tokens.RTE_B_OK, Tokens.RES_EVAC_NO, Tokens.RES_AID_OK, Tokens.RES_ALARM_NO,
                Tokens.PRIO_HIGH, Tokens.CMD_DISP_AID, Tokens.CMD_ROUTE_A, Tokens.CMD_ALARM_OFF, Tokens.CMD_NOTIFY
            ))

            # Scenario 10: Intrusion -> Perimeter alert + Notify
            zone = random.choice([Tokens.ZONE_1, Tokens.ZONE_2, Tokens.ZONE_3, Tokens.ZONE_4])
            samples.append(self._make_sample(
                zone, Tokens.SIG_INTRUSION, Tokens.CONF_MED,
                Tokens.RTE_A_OK, Tokens.RTE_B_OK, Tokens.RES_EVAC_NO, Tokens.RES_AID_NO, Tokens.RES_ALARM_OK,
                Tokens.PRIO_WARNING, Tokens.CMD_ALERT, Tokens.CMD_ROUTE_A, Tokens.CMD_ALARM_ON, Tokens.CMD_NOTIFY
            ))

        random.shuffle(samples)
        return samples[:self.num_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

# ============================================================================
# Part B: Training Pipeline
# ============================================================================
def train_model(model: MicroTransformer, train_loader: DataLoader, epochs: int = 8, lr: float = 1e-3, device: str = 'cpu'):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    total_steps = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            total_steps += 1

            if total_steps % 100 == 0:
                print(f"Step {total_steps} | Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")

    print(f"Training complete. Total steps: {total_steps}")
    return model

# ============================================================================
# Part C: Desktop Evaluation Engine
# ============================================================================
def decode_decision(tokens: List[int]) -> str:
    """Convert output tokens to plain English explanation."""
    if not tokens:
        return "Empty sequence"

    explanations = []
    for t in tokens:
        if t == Tokens.CMD_EVACUATE:
            explanations.append("EVACUATE")
        elif t == Tokens.CMD_MONITOR:
            explanations.append("MONITOR")
        elif t == Tokens.CMD_ALERT:
            explanations.append("ALERT")
        elif t == Tokens.CMD_DISP_AID:
            explanations.append("DISPATCH_AID")
        elif t == Tokens.CMD_ROUTE_A:
            explanations.append("ROUTE_A")
        elif t == Tokens.CMD_ROUTE_B:
            explanations.append("ROUTE_B")
        elif t == Tokens.CMD_ALARM_ON:
            explanations.append("ALARM_ON")
        elif t == Tokens.CMD_ALARM_OFF:
            explanations.append("ALARM_OFF")
        elif t == Tokens.CMD_NOTIFY:
            explanations.append("NOTIFY")
        elif t == Tokens.PRIO_CRITICAL:
            explanations.append("PRIO_CRITICAL")
        elif t == Tokens.PRIO_HIGH:
            explanations.append("PRIO_HIGH")
        elif t == Tokens.PRIO_WARNING:
            explanations.append("PRIO_WARNING")
        elif t == Tokens.PRIO_MONITOR:
            explanations.append("PRIO_MONITOR")

    return " + ".join(explanations) if explanations else "NO_ACTION"

def check_safety_invariants(prompt_tokens: List[int], output_tokens: List[int]) -> Tuple[bool, str]:
    """Verify critical safety invariants."""
    zone = prompt_tokens[1] if len(prompt_tokens) > 1 else 0
    hazard = prompt_tokens[2] if len(prompt_tokens) > 2 else 0
    conf = prompt_tokens[3] if len(prompt_tokens) > 3 else 0
    route_a = prompt_tokens[4] if len(prompt_tokens) > 4 else 0
    route_b = prompt_tokens[5] if len(prompt_tokens) > 5 else 0

    actions = [t for t in output_tokens if t >= 0x80 and t <= 0x8F]
    priorities = [t for t in output_tokens if t >= 0xB0 and t <= 0xB3]
    routes = [t for t in output_tokens if t in (Tokens.CMD_ROUTE_A, Tokens.CMD_ROUTE_B)]
    alarms = [t for t in output_tokens if t in (Tokens.CMD_ALARM_ON, Tokens.CMD_ALARM_OFF)]

    if hazard == Tokens.SIG_ELEC_FIRE:
        if Tokens.PRIO_CRITICAL not in priorities:
            return False, "Fire hazard requires PRIO_CRITICAL"
        if Tokens.CMD_EVACUATE not in actions:
            return False, "Fire hazard requires EVACUATE"
        if Tokens.CMD_ALARM_ON not in alarms:
            return False, "Fire hazard requires ALARM_ON"

    if hazard == Tokens.SIG_LPG_LEAK:
        if Tokens.CMD_ALERT not in actions and Tokens.CMD_EVACUATE not in actions:
            return False, "LPG leak requires ALERT or EVACUATE"
        if Tokens.CMD_ALARM_ON not in alarms:
            return False, "LPG leak requires ALARM_ON"

    if route_a == Tokens.RTE_A_BLOCKED and Tokens.CMD_ROUTE_A in routes:
        return False, "Cannot route to blocked Route A"
    if route_b == Tokens.RTE_B_BLOCKED and Tokens.CMD_ROUTE_B in routes:
        return False, "Cannot route to blocked Route B"

    if conf == Tokens.CONF_LOW and Tokens.PRIO_CRITICAL in priorities:
        return False, "LOW confidence should not yield CRITICAL priority"

    if hazard == Tokens.SIG_MED_REQ:
        if Tokens.CMD_DISP_AID not in actions:
            return False, "Medical request requires DISP_AID"
        if Tokens.CMD_NOTIFY not in actions:
            return False, "Medical request requires NOTIFY"

    if hazard == Tokens.SIG_INTRUSION:
        if Tokens.CMD_ALERT not in actions:
            return False, "Intrusion requires ALERT"
        if Tokens.CMD_NOTIFY not in actions:
            return False, "Intrusion requires NOTIFY"

    return True, "All safety invariants satisfied"

def generate_decision(model: MicroTransformer, prompt_tokens: List[int], device: str = 'cpu') -> List[int]:
    """Autoregressive greedy generation."""
    model.eval()
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        generated = model.generate(prompt_tensor, max_new_tokens=16, temperature=1.0)
    return generated[0].cpu().tolist()

def run_evaluation(model: MicroTransformer, device: str = 'cpu'):
    print("\n" + "="*80)
    print("DESKTOP EVALUATION - 5 HARDCODED TEST SCENARIOS")
    print("="*80)

    test_cases = [
        {
            "name": "Flood - Route A Open",
            "prompt": [Tokens.BOS, Tokens.ZONE_1, Tokens.SIG_FLOOD, Tokens.CONF_HIGH,
                       Tokens.RTE_A_OK, Tokens.RTE_B_OK, Tokens.RES_EVAC_OK, Tokens.RES_AID_NO, Tokens.RES_ALARM_OK, Tokens.SEP],
            "expected": ["EVACUATE", "ROUTE_A", "ALARM_ON", "PRIO_HIGH"]
        },
        {
            "name": "Flood - Route A Blocked",
            "prompt": [Tokens.BOS, Tokens.ZONE_2, Tokens.SIG_FLOOD, Tokens.CONF_HIGH,
                       Tokens.RTE_A_BLOCKED, Tokens.RTE_B_OK, Tokens.RES_EVAC_OK, Tokens.RES_AID_NO, Tokens.RES_ALARM_OK, Tokens.SEP],
            "expected": ["EVACUATE", "ROUTE_B", "ALARM_ON", "PRIO_HIGH"]
        },
        {
            "name": "Electrical Fire - Critical",
            "prompt": [Tokens.BOS, Tokens.ZONE_3, Tokens.SIG_ELEC_FIRE, Tokens.CONF_CRIT,
                       Tokens.RTE_A_OK, Tokens.RTE_B_OK, Tokens.RES_EVAC_OK, Tokens.RES_AID_OK, Tokens.RES_ALARM_OK, Tokens.SEP],
            "expected": ["EVACUATE", "ROUTE_A", "ALARM_ON", "PRIO_CRITICAL"]
        },
        {
            "name": "Weak Sensor Data - Monitor",
            "prompt": [Tokens.BOS, Tokens.ZONE_4, Tokens.SIG_WATER_LEAK, Tokens.CONF_LOW,
                       Tokens.RTE_A_OK, Tokens.RTE_B_OK, Tokens.RES_EVAC_NO, Tokens.RES_AID_NO, Tokens.RES_ALARM_NO, Tokens.SEP],
            "expected": ["MONITOR", "ROUTE_A", "ALARM_OFF", "PRIO_WARNING"]
        },
        {
            "name": "Medical Request - Dispatch Aid",
            "prompt": [Tokens.BOS, Tokens.ZONE_1, Tokens.SIG_MED_REQ, Tokens.CONF_HIGH,
                       Tokens.RTE_A_OK, Tokens.RTE_B_OK, Tokens.RES_EVAC_NO, Tokens.RES_AID_OK, Tokens.RES_ALARM_NO, Tokens.SEP],
            "expected": ["DISPATCH_AID", "ROUTE_A", "ALARM_OFF", "NOTIFY", "PRIO_HIGH"]
        },
    ]

    for i, tc in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {tc['name']} ---")
        print(f"Input:  {tokens_to_hex(tc['prompt'])}")
        print(f"Input:  {tokens_to_names(tc['prompt'])}")

        output = generate_decision(model, tc['prompt'], device)
        sep_idx = output.index(Tokens.SEP) if Tokens.SEP in output else len(output) - 1
        response_tokens = output[sep_idx + 1:]

        print(f"Output: {tokens_to_hex(response_tokens)}")
        print(f"Output: {tokens_to_names(response_tokens)}")
        print(f"Decoded: {decode_decision(response_tokens)}")

        passed, msg = check_safety_invariants(tc['prompt'], response_tokens)
        status = "PASS" if passed else "FAIL"
        print(f"Safety: {status} - {msg}")

# ============================================================================
# Part D: C-Header Exporter
# ============================================================================
def export_c_header(model: MicroTransformer, filepath: str = "model_weights.h"):
    """Export model weights as flat C arrays for ESP32-S3."""
    state_dict = model.state_dict()

    arrays = {}
    shapes = {}

    for name, param in state_dict.items():
        flat = param.detach().cpu().numpy().flatten().astype(np.float32)
        arrays[name] = flat
        shapes[name] = list(param.shape)

    lines = []
    lines.append("#ifndef MODEL_WEIGHTS_H")
    lines.append("#define MODEL_WEIGHTS_H")
    lines.append("")
    lines.append("// Auto-generated model weights for ADSG-Gibber Micro-Transformer")
    lines.append("// Target: Seeed Studio XIAO ESP32-S3 (Xtensa LX7 @ 240 MHz)")
    lines.append(f"// Vocab: {VOCAB_SIZE}, SeqLen: {MAX_SEQ_LEN}, d_model: {D_MODEL}, d_ff: {D_FF}")
    lines.append(f"// Layers: {N_LAYERS}, Heads: {N_HEADS}")
    lines.append("")

    lines.append("#define VOCAB_SIZE 256")
    lines.append("#define MAX_SEQ_LEN 32")
    lines.append("#define D_MODEL 64")
    lines.append("#define D_FF 128")
    lines.append("#define N_LAYERS 3")
    lines.append("#define N_HEADS 4")
    lines.append("#define HEAD_DIM 16")
    lines.append("")

    total_params = 0
    for name, flat in arrays.items():
        total_params += len(flat)
        c_name = name.replace('.', '_').replace('[', '_').replace(']', '')
        lines.append(f"// {name} {shapes[name]} ({len(flat)} elements)")
        lines.append(f"static const float {c_name}[{len(flat)}] = {{")
        for i, val in enumerate(flat):
            if i % 8 == 0:
                lines.append("  ")
            lines.append(f"{val:.8e}f")
            if i < len(flat) - 1:
                lines.append(", ")
            if i % 8 == 7:
                lines.append("\n")
        if len(flat) % 8 != 0:
            lines.append("\n")
        lines.append("};\n")

    lines.append(f"// Total parameters: {total_params}")
    lines.append(f"// Model size: ~{total_params * 4 / 1024:.1f} KB")
    lines.append("")
    lines.append("#endif // MODEL_WEIGHTS_H")

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nExported {total_params} parameters to {filepath}")
    print(f"Model size: ~{total_params * 4 / 1024:.1f} KB")
    return total_params

# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    print("="*80)
    print("ADSG-Gibber Micro-Transformer for XIAO ESP32-S3")
    print("="*80)

    device = 'cpu'
    print(f"Device: {device}")

    # Part A: Dataset
    print("\n[Part A] Generating synthetic dataset (80,000 samples)...")
    dataset = ADSGDataset(num_samples=80000)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=False)
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batches per epoch: {len(train_loader)}")

    x_sample, y_sample = dataset[0]
    print(f"Sample shape: {x_sample.shape}")
    print(f"Sample tokens: {tokens_to_hex(x_sample.tolist())}")
    print(f"Sample names:  {tokens_to_names(x_sample.tolist())}")

    # Part B: Model & Training
    print("\n[Part B] Initializing Micro-Transformer...")
    model = MicroTransformer().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,} ({param_count * 4 / 1024:.1f} KB)")

    print("\nStarting training...")
    model = train_model(model, train_loader, epochs=8, lr=1e-3, device=device)

    # Save checkpoint weights for interactive testing
    checkpoint_path = "adsg_model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nModel weights saved to {checkpoint_path}")

    # Part C: Evaluation
    print("\n[Part C] Running desktop evaluation...")
    run_evaluation(model, device)

    # Part D: Export
    print("\n[Part D] Exporting C header for ESP32-S3...")
    export_c_header(model, "model_weights.h")

    print("\n" + "="*80)
    print("ALL STAGES COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()