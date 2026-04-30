# Continuous Security Assurance (CSA) for IoT Systems

## What is this?

This repository contains the full implementation of my PhD research on Continuous Security Assurance (CSA) for IoT-enabled smart home environments.

Most intrusion detection systems monitor either network traffic or host-level activity, but not both at the same time. That blind spot, which I call the Sys-Net Blind Spot, is what this framework is designed to fix. By fusing system-level and network-level telemetry, the CSA framework produces a continuously updated, interpretable security posture score for each device and for the home as a whole, and it is built to stay reliable even when an attacker is deliberately trying to fool it.

---

## The Core Idea: CSA Constraint-Based Verification

The part of this work I am most proud of is the CSA verification layer. Rather than relying on a machine learning model alone, which can be evaded, every detection goes through a set of physical and semantic consistency checks before it is accepted. The idea is that even if an adversary fools the classifier, they still have to produce behaviour that makes sense in the real world. Often, they cannot.

There are four checks in the verification layer:

**Temporal Consistency** uses EWMA-based statistical baselines to flag behaviour that deviates abnormally over time. This prevents noisy or unstable detections from producing false alarms.

**Causal Consistency** enforces relationships between features, for example the relationship between network traffic and latency. If a change in one feature is not accompanied by a physically plausible change in another, the detection is flagged as inconsistent.

**STRIDE Semantic Consistency** checks that the features associated with a detected attack class are actually present. This catches cases where the ML model is fooled but the resulting behaviour does not actually look like the attack it claims to be.

**Isolation Forest Validation** adds a dual anomaly detection gate as the final decision layer, checking both the source space and the normal space before confirming a detection.

An attack is only confirmed if it passes these checks. Specifically, it is blocked if the physics checks fail and an anomaly is detected, or if dual anomaly confirmation is triggered. This approach reduced adversarial attack success rates from 66.6% to 22.2% in my evaluations, and transfers to external datasets like TON-IoT without any retraining.

---

## How the Pipeline Works

```
IoT Telemetry (System + Network)
        ↓
  Feature Processing
        ↓
ML Detection (Random Forest)
        ↓
  STRIDE Classification
        ↓
CSA Constraint Verification
        ↓
Security Posture Score (0-100)
```

---

## Repository Files

| File | What it does |
|---|---|
| `ton-iot-host-adver.py` | CSA evaluation on TON-IoT host data under adversarial conditions |
| `ton-iot-host-normal.py` | CSA evaluation on TON-IoT host data under normal conditions |
| `ton-iot-network-adver.py` | CSA evaluation on TON-IoT network data under adversarial conditions |
| `ton-iot-network-normal.py` | CSA evaluation on TON-IoT network data under normal conditions |
| `StrideDS.py` | CSA evaluation on the SmartHome-StrideDS dataset |
| `dashboard.py` | Real-time CSA dashboard (Streamlit) |

---

## Datasets

The code is designed to work with two datasets:

- **SmartHome-StrideDS**, a purpose-built dataset I created combining host and network telemetry across four STRIDE attack scenarios. Available on IEEE DataPort (DOI: 10.21227/ty06-es39).
- **TON-IoT**, a publicly available benchmark dataset covering host and network telemetry.

Datasets are not included in this repository due to size. Please download them separately.

---

## Using Your Own Dataset

You do not need to touch the core verification logic. The only things you need to update are in the configuration section at the top of each script:

```python
# Point to your trained model
MODEL_FILE = "your_model.pkl"

# Point to your dataset
DATA_FILE = "your_dataset.csv"

# Update the label column name to match your dataset
TYPE_COL = "type"       # used in TON-IoT scripts
LABEL_COL = "target"    # used in SmartHome scripts

# Update feature lists to match your dataset columns
ALL_FEATURES = [...]
SELECTED_FEATURES = [...]

# Update feature bounds to match your data ranges (important for constraint checks)
FEAT_BOUNDS = {...}
```

Everything else runs as-is.

---

## Installation

```bash
pip install numpy pandas scikit-learn joblib scipy streamlit adversarial-robustness-toolbox plotly
```

---

## Running the System

Run a CSA evaluation:

```bash
python ton-iot-host-adver.py
```

Launch the real-time dashboard:

```bash
streamlit run dashboard.py
```

---

## What the Output Looks Like

Each run produces:

- Attack classification aligned to STRIDE categories
- Adversarial success rate (ASR) before and after CSA verification
- CSA constraint validation results per detection
- A real-time security posture score (0-100) per device and for the environment as a whole

---

## Research Context

This implementation is the core contribution of my PhD thesis:

> *Continuous Security Assurance for IoT-Enabled Smart Homes using Hybrid Telemetry Fusion, Constraint-Based Verification, and Adversarial Robustness*

Birmingham City University, 2022-2026

---

## Author

**Shahbaz Ali Imran**
PhD Researcher, Cybersecurity and AI
Birmingham City University, UK
shahbazimran31@gmail.com

---

## Notes

The code in this repository is the research implementation. Some components are simplified for clarity and reproducibility. Dataset-specific tuning of feature bounds and constraint thresholds will likely be needed if you apply this to a new environment, but the core CSA logic is designed to be dataset-agnostic.
Note: Remove the hostname_check,tamper_file_check,suspicious_port_check,dos_process_check columns while trainig the data on AI algorithm. 
