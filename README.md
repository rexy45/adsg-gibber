# ADSG-Gibber: Sub-150k Parameter Micro-Transformer for Off-Grid Edge Disaster Arbitration

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rexy45/adsg-gibber/blob/main/interactive_demo.ipynb)

An autonomous, edge-native Transformer SLM designed for the **Seeed Studio XIAO ESP32-S3 (Xtensa LX7 dual-core @ 240 MHz)**. ADSG-Gibber operates entirely off-grid without internet connectivity, cloud infrastructure, or external operating systems, arbitrating emergency routing and evacuation directives over an ultra-dense **1-byte custom token protocol**.

---

## Technical Specifications

| Parameter | Specification |
|---|---|
| **Architecture** | 3-Layer Causal Decoder Micro-Transformer |
| **Normalization / Activation** | Pre-Layer RMSNorm, Fast GELU |
| **Total Parameters** | 136,640 parameters (~533.8 KB static C footprint) |
| **Vocabulary Size** | 256 (1-byte dense dialect: `0x00` - `0xFF`) |
| **Sequence Length ($T_{max}$)** | 32 tokens |
| **Hidden Dimension ($d_{model}$)** | 64 |
| **Feed-Forward Dimension ($d_{ff}$)** | 128 |
| **Attention Heads** | 4 heads (Head dimension: 16) |
| **Target Hardware** | Seeed Studio XIAO ESP32-S3 (8 MB Flash, 512 KB SRAM) |
| **Inference Runtime** | Bare-metal static arrays (`model_weights.h`), ~8–15 ms latency |

---

## Token Protocol Architecture

Instead of processing verbose text strings, the model operates on raw 1-byte telemetry packets to maximize bandwidth over low-power radios (ESP-NOW / LoRa):

* **Framing:** `BOS (0x01)`, `EOS (0x02)`, `SEP (0x03)`, `PAD (0x00)`
* **Zone Descriptors:** `ZONE_1 (0x11)` to `ZONE_4 (0x14)`
* **Hazard Telemetry:** `SIG_FLOOD (0x20)`, `SIG_QUAKE (0x23)`, `SIG_ELEC_FIRE (0x24)`, `SIG_LPG_LEAK (0x25)`, etc.
* **Infrastructure State:** `RTE_A_OK (0x40)`, `RTE_A_BLOCKED (0x41)`, `RTE_B_OK (0x42)`, `RTE_B_BLOCKED (0x43)`
* **Confidence Gating:** `CONF_LOW (0x60)` to `CONF_CRIT (0x63)`
* **Edge Directives:** `CMD_EVACUATE (0x80)`, `CMD_ROUTE_A (0x84)`, `CMD_ROUTE_B (0x85)`, `CMD_ALARM_ON (0x86)`

---

## Safety Invariant Verification

The model enforces deterministic crisis invariants across multi-hazard environments:
1. **Critical Fire Escalation:** `SIG_ELEC_FIRE` mandates immediate `PRIO_CRITICAL` + `CMD_EVACUATE` + `CMD_ALARM_ON`.
2. **Dynamic Obstacle Rerouting:** Inputs with `RTE_A_BLOCKED` dynamically re-target evacuation vectors to `CMD_ROUTE_B`.
3. **False Alarm Suppression:** Weak or conflicting sensor readings (`CONF_LOW`) are forced into `CMD_MONITOR` with alarms silenced (`CMD_ALARM_OFF`).

---

## Quick Start & Live Testing

### Option 1: No-Git Direct ZIP Download (Fastest for School Labs)
1. Click the green **Code** button at the top of this repository.
2. Select **Download ZIP** and extract the folder to your desktop.
3. Open Command Prompt or Terminal inside the extracted folder and run:
   ```bash
   pip install -r requirements.txt
   python demo_infer.py
   ### Option 2:
  git clone [https://github.com/rexy45/adsg-gibber.git](https://github.com/rexy45/adsg-gibber.git)
cd adsg-gibber
pip install -r requirements.txt
python demo_infer.py 
