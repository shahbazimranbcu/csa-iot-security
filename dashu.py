"""
CSA Smart Home SOC Dashboard
Shahbaz Ali Imran — PhD Thesis Chapter 7
Birmingham City University — March 2026

Five-Phase Pipeline:
  Phase 1: Hybrid Sys-Net Profiling        (mu, sigma baselines)
  Phase 2: Anomaly Detection               (z-score, A(t), threshold)
  Phase 3: STRIDE IDS Classification       (RF_IDS, F1=96.7%)
  Phase 4: CSA Consistency Verification    (T_EWMA, C_corr, S_STRIDE, IF gates)
  Phase 5: Security Posture Quantification (Risk(t), CSA(t) = 100*exp(-Risk))
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from collections import deque

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CSA Framework — Chapter 7",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# DARK THEME  — force background on every Streamlit layer
# ─────────────────────────────────────────────────────────────────────
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Orbitron:wght@700;900&family=Exo+2:wght@400;600;700&display=swap');

html, body { background: #04090f !important; }

/* every Streamlit wrapper */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
.main,
.main .block-container,
[data-testid="stVerticalBlock"],
[data-testid="column"],
div.element-container,
div.stMarkdown,
div.stPlotlyChart { background: #04090f !important; color: #8aaccc !important; }

.main .block-container { padding: 0.4rem 1.1rem 2rem !important; }

section[data-testid="stSidebar"] {
    background: #020609 !important;
    border-right: 1px solid #091828 !important;
}
section[data-testid="stSidebar"] * { color: #4a7a9a !important; }

/* buttons */
.stButton > button {
    background: #040d1a !important;
    color: #2a7fff !important;
    border: 1px solid #0d2a4a !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .71rem !important;
    border-radius: 2px !important;
}
.stButton > button:hover {
    border-color: #2a7fff !important;
    background: #061428 !important;
}

/* text */
p, label, div, span, li {
    color: #8aaccc !important;
    font-family: 'Exo 2', sans-serif !important;
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ── header ─────────────────────────── */
.hdr {
    background: linear-gradient(135deg, #020810 0%, #050f1e 55%, #030b18 100%);
    border-top: 2px solid #0044bb;
    border-bottom: 1px solid #091828;
    padding: 14px 22px 12px;
    margin: -0.4rem -1.1rem 0.7rem;
}
.hdr-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.0rem; font-weight: 900;
    color: #1a6eee; letter-spacing: 3px; margin: 0;
}
.hdr-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: .6rem; color: #1a3a5a; letter-spacing: 1.5px; margin-top: 4px;
}

/* ── 5-phase pipeline ────────────────── */
.pipe { display: flex; gap: 2px; margin: .4rem 0 .7rem; }
.pbox {
    flex: 1; background: #040d1a;
    border: 1px solid #08172a; border-top: 2px solid #08172a;
    border-radius: 2px; padding: 7px 7px 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: .53rem; color: #122030; line-height: 1.65;
}
.pbox.on { border-top-color: #0044bb; color: #1e5a88; background: #050e1c; }
.pnum {
    font-family: 'Orbitron', monospace;
    font-size: .82rem; color: #0a1e34; margin-bottom: 2px;
}
.pbox.on .pnum { color: #0044bb; }

/* ── banner ──────────────────────────── */
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.25} }
.bnr {
    padding: 7px 14px; border-radius: 2px; border-left: 3px solid;
    font-family: 'JetBrains Mono', monospace;
    font-size: .7rem; letter-spacing: 1.5px; margin: .4rem 0 .6rem;
}
.bnr-ok  { background: #020a06; border-color: #009944; color: #009944; }
.bnr-std { background: #0a0306; border-color: #bb1111; color: #bb1111;
           animation: blink .8s infinite; }
.bnr-adv { background: #070210; border-color: #7722bb; color: #7722bb;
           animation: blink .5s infinite; }

/* ── metric tile ─────────────────────── */
.tile {
    background: #040c18; border: 1px solid #081622;
    border-left: 2px solid #091828; border-radius: 2px; padding: 9px 11px;
}
.tlabel {
    font-family: 'JetBrains Mono', monospace; font-size: .54rem;
    color: #112030; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 2px;
}
.tval {
    font-family: 'Orbitron', monospace;
    font-size: 1.45rem; font-weight: 900; line-height: 1.1;
}
.tsub {
    font-family: 'JetBrains Mono', monospace;
    font-size: .53rem; color: #112030; margin-top: 3px;
}
.c0 { color: #009944; border-left-color: #009944 !important; }
.c1 { color: #1a77ee; border-left-color: #1a77ee !important; }
.c2 { color: #cc1111; border-left-color: #cc1111 !important; }
.c3 { color: #bb55ff; border-left-color: #bb55ff !important; }
.c4 { color: #ee9900; border-left-color: #ee9900 !important; }

/* ── device card ─────────────────────── */
.dcard {
    background: #050c18; border: 1px solid #081622;
    border-radius: 2px; padding: 8px 10px; margin-bottom: 4px;
}
.dcard.atk { border-color: #2a0606; background: #070606; }
.dname {
    font-family: 'JetBrains Mono', monospace;
    font-size: .6rem; color: #122030; margin-bottom: 3px;
}
.dcard.atk .dname { color: #bb1111; }
.dscore { font-family: 'Orbitron', monospace; font-size: 1.1rem; font-weight: 900; }
.dfeats {
    font-family: 'JetBrains Mono', monospace; font-size: .52rem;
    color: #0e1e2e; margin-top: 4px; line-height: 1.7;
}
.dfeats span  { color: #1a3a54; }
.dfeats .hi   { color: #cc3311 !important; }
.dgates { display: flex; flex-wrap: wrap; gap: 2px; margin-top: 3px; }
.g {
    font-family: 'JetBrains Mono', monospace; font-size: .51rem;
    padding: 1px 5px; border-radius: 1px;
}
.goff  { background: #050d18; color: #0c1e2e; border: 1px solid #081622; }
.gewma { background: #1a0000; color: #ff3333; border: 1px solid #380000; }
.gcorr { background: #180800; color: #ff7722; border: 1px solid #381400; }
.gstrd { background: #171700; color: #ddbb00; border: 1px solid #343200; }
.gif   { background: #00001a; color: #2255ee; border: 1px solid #001440; }

/* ── equation panel ──────────────────── */
.eqp {
    background: #040c18; border: 1px solid #152b42; /* BRIGHTENED BORDER */
    border-radius: 2px; padding: 8px 11px; margin-bottom: 4px;
}
.eqt {
    font-family: 'JetBrains Mono', monospace; font-size: .56rem;
    color: #6bb0e8; /* BRIGHTENED TEXT */
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 4px;
}
.eqb {
    font-family: 'JetBrains Mono', monospace;
    font-size: .66rem; color: #a9cce3; /* BRIGHTENED TEXT */
    line-height: 1.9;
}
.eqv { color: #009944; } .eqw { color: #ee9900; } .eqe { color: #cc1111; }

/* ── section heading ─────────────────── */
.sh {
    font-family: 'JetBrains Mono', monospace; font-size: .58rem;
    color: #122030; letter-spacing: 3px; text-transform: uppercase;
    border-bottom: 1px solid #081622; padding-bottom: 4px; margin: .9rem 0 .5rem;
}

/* ── event log ───────────────────────── */
.elog {
    background: #030a14; border: 1px solid #081622;
    border-radius: 2px; padding: 7px 11px;
    max-height: 160px; overflow-y: auto;
    font-family: 'JetBrains Mono', monospace; font-size: .58rem; line-height: 1.9;
}

/* ── footer ──────────────────────────── */
.foot {
    font-family: 'JetBrains Mono', monospace; font-size: .51rem; color: #0b1a28;
    border-top: 1px solid #081622; margin-top: 1.2rem; padding-top: .5rem; text-align: center;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# THESIS CONSTANTS  (SmartHome-StrideDS, §7.3–7.4)
# ─────────────────────────────────────────────────────────────────────
EPS   = 1e-9
Z_MAX = 5.0          # cap on z-score contribution  (Eq 7.11)
TAU_Z = 3.5          # EWMA z-threshold  (Eq 7.3)

# Per-feature bounds (used for causal delta normalisation)
FEAT_BOUNDS = {
    "cpu": (0., 100.), "memory": (5.99, 64.46), "disk": (68., 95.),
    "load": (0., 28.99), "tcp_conn": (2., 45.),
    "icmp_packets": (2., 134562.), "latency": (0., 6182.53),
}

# Per-class baselines  μ_j , σ_j   (Table 4.3, §4.5)
# Each entry: (mean, std)
BL = {
    "Normal":   dict(cpu=(12.,5.), memory=(22.,6.), disk=(73.,3.),
                     load=(.35,.2), tcp_conn=(5.,2.),
                     icmp_packets=(3.,1.5), latency=(11.,4.)),
    "DOS":      dict(cpu=(38.5,18.), memory=(48.,11.), disk=(75.,4.5),
                     load=(8.7,4.), tcp_conn=(28.,9.),
                     icmp_packets=(51709,22000), latency=(340.,145.)),
    "Spoofing": dict(cpu=(18.,9.), memory=(27.,8.5), disk=(73.5,4.2),
                     load=(.55,.35), tcp_conn=(12.,5.5),
                     icmp_packets=(8200,3900), latency=(820.,280.)),
    "Tampering":dict(cpu=(72.,19.), memory=(56.5,12.), disk=(78.5,5.),
                     load=(9.8,4.5), tcp_conn=(8.,3.8),
                     icmp_packets=(5.,2.8), latency=(28.,11.)),
    "Info_Disclosure": dict(
                     cpu=(34.,14.), memory=(38.5,10.5), disk=(82.,6.5),
                     load=(3.1,1.8), tcp_conn=(14.5,6.),
                     icmp_packets=(6800,3200), latency=(95.,42.)),
}

# Wide posture baseline σ — Eq 7.10–7.12
# Operational tolerance band: benign z ≈ 0.03–0.07 → Risk ≈ 0 → CSA ≈ 98–100
PSIG = dict(cpu=40., memory=25., disk=15., load=3.,
            tcp_conn=20., icmp_packets=1000., latency=50.)
PMU  = {k: BL["Normal"][k][0] for k in BL["Normal"]}

# Feature security-relevance weights  w_j  (§7.4, Eq 7.11)
W = dict(cpu=.20, memory=.15, disk=.10, load=.15,
         tcp_conn=.15, icmp_packets=.15, latency=.10)

# STRIDE signature features per class  (Eq 7.6)
STRIDE_FEATS = dict(
    DOS=["icmp_packets", "tcp_conn"],
    Spoofing=["icmp_packets", "latency"],
    Tampering=["cpu", "memory", "load"],
    Info_Disclosure=["cpu", "disk"],
)

DEVICES    = [f"IoT-{i:02d}" for i in range(1, 10)]
ATTACK_MAP = {1: "DOS", 6: "Tampering", 3: "Spoofing", 4: "Info_Disclosure"}

# ── Timeline colours — plain strings, NO string manipulation ──────────
LINE_C = {
    "Normal Traffic":     "#009944",
    "Standard Attack":    "#cc1111",
    "Adversarial Attack": "#7722bb",
}
FILL_C = {
    "Normal Traffic":     "rgba(0,153,68,0.07)",
    "Standard Attack":    "rgba(204,17,17,0.07)",
    "Adversarial Attack": "rgba(119,34,187,0.07)",
}


# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────
def _init():
    defaults = dict(
        running=False, mode="Normal Traffic", tick=0,
        history=pd.DataFrame(columns=["Tick", "Score", "Mode"]),
        event_log=deque(maxlen=200),
        metrics=dict(total=0, ml=0, csa=0, adv=0, adv_ml=0, adv_csa=0),
        seed=42,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()


# ─────────────────────────────────────────────────────────────────────
# TELEMETRY GENERATORS
# ─────────────────────────────────────────────────────────────────────
def _clip(val, lo, hi):
    return float(np.clip(val, lo, hi))

def gen_normal(rng):
    b = BL["Normal"]
    return {
        "cpu":          _clip(rng.normal(b["cpu"][0],          b["cpu"][1]  * .35), .5,  20.),
        "memory":       _clip(rng.normal(b["memory"][0],       b["memory"][1]*.35), 10., 35.),
        "disk":         _clip(rng.normal(b["disk"][0],         b["disk"][1] * .35), 68., 80.),
        "load":         _clip(rng.normal(b["load"][0],         b["load"][1] * .35), .05, .65),
        "tcp_conn":     int(  np.clip(rng.normal(b["tcp_conn"][0],     b["tcp_conn"][1]*.35), 2, 9)),
        "icmp_packets": int(  np.clip(rng.normal(b["icmp_packets"][0], b["icmp_packets"][1]*.35), 1, 5)),
        "latency":      _clip(rng.normal(b["latency"][0],      b["latency"][1]* .35), 2., 20.),
    }

# Standard attack params  (mu, sig, lo, hi)
_S = {
    "DOS": dict(
        cpu=(38,12,20,65), memory=(48,8,35,62), disk=(75,3,68,82),
        load=(8.7,3.,4.,18.), tcp_conn=(130,15,80,160),
        icmp_packets=(51709,8000,30000,90000), latency=(340,80,200,700)),
    "Tampering": dict(
        cpu=(85,8,70,99), memory=(72,6,60,85), disk=(79,4,68,90),
        load=(14.,4.,8.,22.), tcp_conn=(8,3,2,18),
        icmp_packets=(5,2,2,12), latency=(28,10,10,55)),
    "Spoofing": dict(
        cpu=(18,7,5,35), memory=(27,7,15,42), disk=(74,4,68,84),
        load=(.55,.3,.1,1.5), tcp_conn=(12,5,4,28),
        icmp_packets=(8200,1500,4000,14000), latency=(820,100,500,1200)),
    "Info_Disclosure": dict(
        cpu=(34,10,15,55), memory=(38,8,22,54), disk=(82,5,72,93),
        load=(3.1,1.2,.5,7.), tcp_conn=(14,5,6,28),
        icmp_packets=(6800,1200,3000,12000), latency=(95,30,40,200)),
}

# Adversarial attack params  (ML fooled, CSA physics catches)
_A = {
    "DOS": dict(
        cpu=(18,5,5,35), memory=(26,6,14,40), disk=(74,3,68,82),
        load=(4.59,1.,2.,9.), tcp_conn=(5,2,2,12),
        icmp_packets=(31525,5000,18000,50000), latency=(216,60,80,420)),
    "Tampering": dict(
        cpu=(82,6,68,98), memory=(24,5,14,36), disk=(75,3,68,82),
        load=(.5,.3,0.,1.5), tcp_conn=(7,3,2,14),
        icmp_packets=(4,2,2,10), latency=(15,6,5,35)),
    "Spoofing": dict(
        cpu=(10,4,3,22), memory=(24,6,13,38), disk=(73,3,68,80),
        load=(.3,.2,0.,1.), tcp_conn=(3,1,2,6),
        icmp_packets=(2800,800,1200,5500), latency=(904,80,700,1100)),
    "Info_Disclosure": dict(
        cpu=(.5,.4,0.,2.), memory=(24,6,13,38), disk=(83,4,72,92),
        load=(.2,.2,0.,.9), tcp_conn=(9,4,2,20),
        icmp_packets=(2,1,2,6), latency=(602,80,450,820)),
}

def _draw(p, rng, k):
    m, s, lo, hi = p[k]
    v = rng.normal(m, s)
    if isinstance(m, float) or isinstance(lo, float):
        return _clip(v, lo, hi)
    return int(np.clip(v, lo, hi))

def gen_attack(cls, params, rng):
    return {k: _draw(params[cls], rng, k)
            for k in ["cpu","memory","disk","load","tcp_conn","icmp_packets","latency"]}


# ─────────────────────────────────────────────────────────────────────
# PHASE 2 — Anomaly Score   A(t) = Σ w_i * |f_i − μ_i| / σ_i
# ─────────────────────────────────────────────────────────────────────
def phase2_anomaly(f):
    b = BL["Normal"]
    A = sum(W[k] * min(abs(f[k] - b[k][0]) / (b[k][1] + EPS), Z_MAX) for k in f)
    return A > 1.2, round(A, 3)


# ─────────────────────────────────────────────────────────────────────
# PHASE 4 — CSA Consistency Gates
# ─────────────────────────────────────────────────────────────────────
def gate_ewma(f, cls):
    """Eq 7.2–7.3  T_EWMA gate"""
    b = BL.get(cls, BL["Normal"])
    viols = sum(1 for k in f
                if abs(f[k] - b[k][0]) / (b[k][1] * .9 + EPS) > TAU_Z)
    return viols >= (len(f) // 2), viols

def gate_corr(orig, adv):
    """Eq 7.4–7.5  C_corr gate"""
    rules = [("latency","tcp_conn",.05),
             ("cpu","load",.10),
             ("icmp_packets","latency",.0001)]
    fired = []
    for fA, fB, ratio in rules:
        span = FEAT_BOUNDS[fA][1] - FEAT_BOUNDS[fA][0]
        dA = adv[fA] - orig[fA];  dB = adv[fB] - orig[fB]
        if abs(dA) > .01 * span:
            if (dA > 0 and dB < ratio * dA) or (dA < 0 and dB > ratio * dA):
                fired.append(f"{fA}→{fB}")
    return len(fired) >= 3, len(fired), fired

def gate_stride(f, cls):
    """Eq 7.6  S_STRIDE gate"""
    if cls not in STRIDE_FEATS:
        return False, 0
    b = BL[cls]
    elev = sum(1 for feat in STRIDE_FEATS[cls]
               if f.get(feat, 0) >= b[feat][0] * 1.5)
    return elev == 0, elev

def gate_if(f, cls):
    """Eq 7.7  Isolation Forest confirmation gates (simulated)"""
    def dist(feat, c):
        b = BL[c]
        return np.sqrt(sum(((feat[k] - b[k][0]) / (b[k][1] + EPS)) ** 2
                           for k in feat) / len(feat))
    return dist(f, cls) > 4.5, dist(f, "Normal") > 4.5

def csa_master_gate(orig, adv, cls):
    """
    Eq 7.8–7.9  Master detection gate
    Block = (Phi AND IF_src) OR (Phi AND IF_norm) OR (IF_src AND IF_norm)
    where Phi = T_EWMA_fails OR C_corr_fails OR S_STRIDE_fails
    """
    ef, ev  = gate_ewma(adv, cls)
    cf, cv, cr = gate_corr(orig, adv)
    sf, se  = gate_stride(adv, cls)
    src, nrm = gate_if(adv, cls)
    phi = ef or cf or sf
    blocked = (phi and src) or (phi and nrm) or (src and nrm)
    reasons = (
        (["T_EWMA"]   if ef  else []) +
        (["C_corr"]   if cf  else []) +
        (["S_STRIDE"] if sf  else []) +
        (["IF_src"]   if src else []) +
        (["IF_norm"]  if nrm else [])
    )
    cond = ("III" if (src and nrm) else
            "II"  if (phi and nrm) else "I")
    return blocked, reasons, dict(
        ewma=(ef, ev), corr=(cf, cv), stride=(sf, se),
        src=src, nrm=nrm, cond=cond
    )


# ─────────────────────────────────────────────────────────────────────
# PHASE 5 — Posture Score   CSA(t) = 100 * exp(−Risk(t))
# Eq 7.10:  z_j(t) = |F_j(t) − μ_j| / (σ_j + ε)
# Eq 7.11:  Risk(t) = Σ w_j * min(z_j, z_max)
# Eq 7.12:  CSA(t)  = 100 * exp(−Risk(t))
# ─────────────────────────────────────────────────────────────────────
def posture_score(feats_list):
    scores = []
    for f in feats_list:
        risk = sum(W[k] * min(abs(f[k] - PMU[k]) / (PSIG[k] + EPS), Z_MAX)
                   for k in f)
        scores.append(max(0., min(100., 100. * np.exp(-risk))))
    return min(scores) if scores else 100.


# ─────────────────────────────────────────────────────────────────────
# MAIN SIMULATION TICK
# ─────────────────────────────────────────────────────────────────────
def run_tick(mode):
    rng = np.random.RandomState(st.session_state.seed + st.session_state.tick)
    rows = [];  all_f = []
    m = st.session_state.metrics

    for i, dev in enumerate(DEVICES):
        is_atk = (mode != "Normal Traffic") and (i in ATTACK_MAP)
        cls    = ATTACK_MAP.get(i) if is_atk else None

        # Phase 1 — generate telemetry
        if not is_atk:
            f  = gen_normal(rng)
            ml = "Normal"
        elif mode == "Standard Attack":
            f  = gen_attack(cls, _S, rng)
            ml = cls
        else:
            f  = gen_attack(cls, _A, rng)
            ml = "Normal"   # ML is fooled
        all_f.append(f)

        # Phase 2 — anomaly detection
        is_anom, A_t = phase2_anomaly(f)
        if not is_anom:
            ml = "Normal"

        # Phase 4 — CSA gate
        blocked = False;  reasons = [];  gates = {};  verdict = "PASS ✓"
        if is_atk:
            m["total"] += 1
            if mode == "Adversarial Attack":
                m["adv"] += 1
            if ml == cls:
                m["ml"] += 1
            blocked, reasons, gates = csa_master_gate(f, f, cls)
            if blocked:
                m["csa"] += 1
                if mode == "Adversarial Attack" and ml == "Normal":
                    m["adv_ml"] += 1
                    m["adv_csa"] += 1
                verdict = "BLOCKED 🔒"
            else:
                verdict = "EVADED ⚠️"

        # Phase 5 — per-device posture
        dev_risk = sum(W[k] * min(abs(f[k] - PMU[k]) / (PSIG[k] + EPS), Z_MAX)
                       for k in f)
        dev_csa = max(0., min(100., 100. * np.exp(-dev_risk)))

        rows.append(dict(
            dev=dev, cls=cls or "Normal", is_atk=is_atk,
            f=f, A_t=A_t, ml=ml,
            blocked=blocked, reasons=reasons, gates=gates,
            verdict=verdict, dev_csa=round(dev_csa, 1),
        ))

        if is_atk:
            st.session_state.event_log.appendleft(dict(
                t=st.session_state.tick,
                tag="ADV" if mode == "Adversarial Attack" else "STD",
                dev=dev, cls=cls,
                ml_miss=(ml == "Normal"), blocked=blocked,
                reasons=reasons,
                cond=gates.get("cond", ""),
            ))

    overall = posture_score(all_f)
    st.session_state.tick += 1
    new = pd.DataFrame({"Tick": [st.session_state.tick],
                         "Score": [round(overall, 1)],
                         "Mode":  [mode]})
    st.session_state.history = pd.concat(
        [st.session_state.history, new]).tail(100)
    return rows, overall


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**CSA SOC CONTROL**")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ RUN",  use_container_width=True):
            st.session_state.running = True
    with c2:
        if st.button("⏸ STOP", use_container_width=True):
            st.session_state.running = False
    st.markdown("---")
    st.markdown("**Inject scenario:**")
    if st.button("🟢  Normal Traffic",     use_container_width=True):
        st.session_state.mode = "Normal Traffic"
    if st.button("🔴  Standard Attack",    use_container_width=True):
        st.session_state.mode = "Standard Attack"
    if st.button("🟣  Adversarial Attack", use_container_width=True):
        st.session_state.mode = "Adversarial Attack"
    st.markdown("---")
    m = st.session_state.metrics
    st.markdown(f"""
**Framework:** CSA (§7.2)

Eq 7.3: τ_z = {TAU_Z}  
Eq 7.11: z_max = {Z_MAX}  
Eq 7.12: CSA = 100·e^(−Risk)  
Dataset: SmartHome-StrideDS  
RF F1: 96.7%  TPR: 99.83%

**Session stats:** Total attacks : {m['total']}  
ML caught     : {m['ml']}  
CSA blocked   : {m['csa']}  
Adv attempts  : {m['adv']}  
Adv CSA saved : {m['adv_csa']}
""")
    st.markdown("---")
    if st.button("🔄  Reset", use_container_width=True):
        for k in ["history", "event_log", "metrics", "tick"]:
            del st.session_state[k]
        _init()
        st.session_state.mode    = "Normal Traffic"
        st.session_state.running = False
        st.rerun()


# ─────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────
mode = st.session_state.mode
mc   = ("#009944" if mode == "Normal Traffic" else
        "#cc1111" if mode == "Standard Attack" else "#7722bb")

st.markdown(f"""
<div class="hdr">
  <div class="hdr-title">
    🛡️ IoT Smart Home — Continuous Security Assurance Framework
  </div>
  <div class="hdr-sub">
    PhD Work &nbsp;·&nbsp; Shahbaz Ali Imran &nbsp;·&nbsp;
    Birmingham City University 2026 &nbsp;·&nbsp; SmartHome-StrideDS
    &nbsp;·&nbsp; RF F1 = 96.7% &nbsp;·&nbsp;
    TICK <span style="color:#1a6eee;font-family:'Orbitron',monospace">
      {st.session_state.tick:04d}</span>
    &nbsp;·&nbsp;
    MODE <span style="color:{mc}">{mode.upper()}</span>
  </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# FIVE-PHASE PIPELINE BAR
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pipe">
  <div class="pbox on">
    <div class="pnum">①</div>
    SYS-NET PROFILING<br>
    μ_j σ_j per device<br>
    Eq 4.1–4.3  §4.4<br>
    X(t)={F₁…Fₙ}  Eq 7.1
  </div>
  <div class="pbox on">
    <div class="pnum">②</div>
    ANOMALY DETECTION<br>
    A_i=|f_i−μ_i|/σ_i<br>
    A(t)=Σwᵢ·Aᵢ  Eq 5.3<br>
    RF_AD binary  §5.4
  </div>
  <div class="pbox on">
    <div class="pnum">③</div>
    STRIDE IDS<br>
    RF_IDS multi-class<br>
    S·T·D·I  §6.4<br>
    F1=96.7%  Table 6.1
  </div>
  <div class="pbox on">
    <div class="pnum">④</div>
    CSA PHYSICS GATE<br>
    T_EWMA  Eq 7.2–7.3<br>
    C_corr  Eq 7.4–7.5<br>
    S_STRIDE  Eq 7.6<br>
    IF  Eq 7.7  Gate 7.8
  </div>
  <div class="pbox on">
    <div class="pnum">⑤</div>
    POSTURE SCORE<br>
    z_j=|F_j−μ_j|/(σ_j+ε)<br>
    Risk=Σwⱼ·min(zⱼ,z_max)<br>
    CSA=100·e^(−Risk)  7.12
  </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────
if mode == "Normal Traffic":
    st.markdown(
        "<div class='bnr bnr-ok'>✔  NOMINAL OPERATION — "
        "ALL DEVICES WITHIN BEHAVIOURAL BASELINE — CSA(t) ≈ 98–100</div>",
        unsafe_allow_html=True)
elif mode == "Standard Attack":
    st.markdown(
        "<div class='bnr bnr-std'>⚠  ACTIVE THREAT — "
        "DOS · TAMPERING · SPOOFING · INFO_DISCLOSURE — ML + CSA ENGAGED</div>",
        unsafe_allow_html=True)
else:
    st.markdown(
        "<div class='bnr bnr-adv'>⚡  ADVERSARIAL BLACK-BOX ATTACK — "
        "RF_IDS BOUNDARY EVADED — CSA PHYSICS GATE INTERCEPTING</div>",
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# RUN TICK
# ─────────────────────────────────────────────────────────────────────
rows, overall = run_tick(mode)
m       = st.session_state.metrics
total_a = m["total"]
ml_r    = (m["ml"]      / total_a * 100) if total_a else 100.
csa_r   = (m["csa"]     / total_a * 100) if total_a else 100.
adv_r   = (m["adv_csa"] / m["adv"] * 100) if m["adv"] else 100.


# ─────────────────────────────────────────────────────────────────────
# ROW 1 — GAUGE + TILES
# ─────────────────────────────────────────────────────────────────────
gc, t1, t2, t3, t4, t5 = st.columns([2, 1, 1, 1, 1, 1])

hc = ("#009944" if overall > 75 else
      "#ee9900" if overall > 40 else "#cc1111")

with gc:
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(overall, 1),
        delta=dict(reference=98.,
                   decreasing=dict(color="#cc1111"),
                   increasing=dict(color="#009944"),
                   font=dict(size=13, family="JetBrains Mono")),
        number=dict(suffix="%",
                    font=dict(color=hc, family="Orbitron", size=38)),
        title=dict(text="CSA(t) = 100 · e^(−Risk(t))  Eq 7.12",
                   font=dict(size=10, color="#1a3a5a", family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[0, 100],
                      tickfont=dict(color="#0c1e30", size=8,
                                    family="JetBrains Mono"),
                      tickcolor="#081622"),
            bar=dict(color=hc, thickness=0.2),
            bgcolor="#040c18", borderwidth=0,
            steps=[
                dict(range=[0,  40], color="rgba(204,17,17,.05)"),
                dict(range=[40, 75], color="rgba(238,153,0,.05)"),
                dict(range=[75,100], color="rgba(0,153,68,.05)"),
            ],
            threshold=dict(line=dict(color="#1144aa", width=2),
                           thickness=.75, value=75),
        )
    ))
    fig_g.update_layout(
        height=215,
        margin=dict(l=10, r=10, t=50, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8aaccc"),
    )
    st.plotly_chart(fig_g, use_container_width=True,
                    key=f"g_{st.session_state.tick}")

def _tile(col, label, val, unit, sub, css):
    col.markdown(
        f'<div class="tile {css}">'
        f'<div class="tlabel">{label}</div>'
        f'<div class="tval">{val}'
        f'<span style="font-size:.82rem;opacity:.55">{unit}</span></div>'
        f'<div class="tsub">{sub}</div>'
        f'</div>', unsafe_allow_html=True)

_tile(t1, "ML IDS Catch",  f"{ml_r:.0f}", "%", "RF_IDS §6.4",
      "c1" if ml_r  > 80 else "c4")
_tile(t2, "CSA Block Rate", f"{csa_r:.0f}", "%", "Eq 7.8–7.9",
      "c0" if csa_r > 90 else "c4")
_tile(t3, "Adv Block Rate", f"{adv_r:.0f}", "%", "Ch.8 §8.5", "c3")
_tile(t4, "Total Attacks",  str(total_a),   "",  "this session",
      "c2" if total_a > 0 else "c1")
_tile(t5, "Adv CSA Saved",  str(m["adv_csa"]), "", "Ch.8 result", "c0")


# ─────────────────────────────────────────────────────────────────────
# ROW 2 — DEVICE GRID  (3 × 3)
# ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="sh">LIVE DEVICE TELEMETRY — '
    'PHASES 1–4  (Sys-Net Hybrid Features)</div>',
    unsafe_allow_html=True)

G_KEYS = [
    ("T_EWMA",   "ewma",  "gewma"),
    ("C_corr",   "corr",  "gcorr"),
    ("S_STRIDE", "stride","gstrd"),
    ("IF_src",   "src",   "gif"),
    ("IF_norm",  "nrm",   "gif"),
]

cols3 = st.columns(3)
for idx, r in enumerate(rows):
    with cols3[idx % 3]:
        sc = ("#009944" if r["dev_csa"] > 75 else
              "#ee9900" if r["dev_csa"] > 40 else "#cc1111")
        atk = "atk" if r["is_atk"] else ""
        f   = r["f"]

        # gate badges
        gh = "<div class='dgates'>"
        for label, key, css in G_KEYS:
            fired = label in r["reasons"]
            gh += (f"<span class='g {css}'>{label}</span>"
                   if fired else
                   f"<span class='g goff'>{label}</span>")
        gh += "</div>"

        # feature values
        fh = (
            f"<div class='dfeats'>"
            f"cpu:<span{' class=hi' if f['cpu']>50 else ''}>"
            f"{f['cpu']:.1f}</span>% "
            f"mem:<span>{f['memory']:.1f}</span>% "
            f"load:<span{' class=hi' if f['load']>5 else ''}>"
            f"{f['load']:.2f}</span><br>"
            f"icmp:<span{' class=hi' if f['icmp_packets']>20000 else ''}>"
            f"{f['icmp_packets']:,}</span> "
            f"lat:<span{' class=hi' if f['latency']>300 else ''}>"
            f"{f['latency']:.0f}</span>ms "
            f"A(t):<span{' class=hi' if r['A_t']>1.2 else ''}>"
            f"{r['A_t']:.3f}</span>"
            f"</div>"
        )

        vc = ("#cc1111" if "BLOCKED" in r["verdict"] else
              "#ee7700" if "EVADED"  in r["verdict"] else "#009944")
        tag = f" [{r['cls']}]" if r["is_atk"] else ""
        
        # HTML/SVG Dynamic Circular Gauge
        dash = (r['dev_csa'] / 100.0) * 125.66
        gauge_html = (
            f"<div style='position:relative; width:45px; height:30px; text-align:center;'>"
            f"<svg viewBox='0 0 100 55' width='100%' height='100%'>"
            f"<path d='M 10 50 A 40 40 0 0 1 90 50' fill='none' stroke='#081622' stroke-width='12' stroke-linecap='round'/>"
            f"<path d='M 10 50 A 40 40 0 0 1 90 50' fill='none' stroke='{sc}' stroke-width='12' stroke-linecap='round' stroke-dasharray='{dash}, 126' />"
            f"</svg>"
            f"<div class='dscore' style='position:absolute; bottom:-6px; width:100%; color:{sc}; font-size:.85rem; line-height:1;'>"
            f"{r['dev_csa']:.0f}<span style='font-size:.45rem;opacity:.45'>%</span>"
            f"</div>"
            f"</div>"
        )

        st.markdown(f"""
<div class="dcard {atk}">
  <div class="dname">{'⚠ ' if r['is_atk'] else '● '}{r['dev']}{tag}</div>
  <div style="display:flex;justify-content:space-between;align-items:center;">
{gauge_html}
    <div style="font-family:'JetBrains Mono',monospace; font-size:.58rem;color:{vc}">{r['verdict']}</div>
  </div>
  <div style="font-family:'JetBrains Mono',monospace; font-size:.54rem;color:#0e1e2c;margin-top:2px">
    Ph3:<span style="color:#1a4060">&nbsp;{r['ml']}</span>
    &nbsp; Ph4:<span style="color:{vc}">&nbsp;{'BLOCKED' if r['blocked'] else ('PASS' if r['is_atk'] else '—')}</span>
  </div>
  {fh}{gh}
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# ROW 3 — POSTURE TIMELINE  |  LIVE EQUATIONS  |  STRIDE RADAR
# ─────────────────────────────────────────────────────────────────────
tc, ec, rc = st.columns([3, 1.6, 1.4])

# ── Timeline ──────────────────────────────────────────────────────────
with tc:
    st.markdown(
        '<div class="sh">POSTURE TIMELINE — '
        'CSA(t) = 100 · e^(−Risk(t))  (Eq 7.12, §7.4)</div>',
        unsafe_allow_html=True)
    hist = st.session_state.history
    if not hist.empty:
        fig_l = go.Figure()
        for mn, grp in hist.groupby("Mode"):
            lc = LINE_C.get(mn, "#1a77ee")
            fc = FILL_C.get(mn, "rgba(26,119,238,0.07)")
            fig_l.add_trace(go.Scatter(
                x=grp["Tick"], y=grp["Score"],
                mode="lines", name=mn,
                line=dict(color=lc, width=2),
                fill="tozeroy",
                fillcolor=fc,
            ))
        fig_l.add_hline(
            y=75, line_dash="dot", line_color="#6bb0e8", # BRIGHTENED
            annotation_text="Healthy ≥ 75",
            annotation_font=dict(color="#6bb0e8", size=9, # BRIGHTENED
                                  family="JetBrains Mono"))
        fig_l.update_layout(
            yaxis=dict(
                range=[0, 105], gridcolor="#152535", # BRIGHTENED
                title=dict(text="CSA(t)",
                           font=dict(color="#8aaccc", size=10, # BRIGHTENED
                                     family="JetBrains Mono")),
                tickfont=dict(color="#8aaccc", family="JetBrains Mono"), # BRIGHTENED
            ),
            xaxis=dict(
                gridcolor="#152535", # BRIGHTENED
                title=dict(text="Monitoring ticks",
                           font=dict(color="#8aaccc", size=10, # BRIGHTENED
                                     family="JetBrains Mono")),
                tickfont=dict(color="#8aaccc", family="JetBrains Mono"), # BRIGHTENED
            ),
            height=280, margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8aaccc", family="JetBrains Mono"),
            legend=dict(font=dict(size=9, color="#8aaccc", # BRIGHTENED
                                   family="JetBrains Mono"),
                        bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_l, use_container_width=True,
                        key=f"l_{st.session_state.tick}")

# ── Live Equations ────────────────────────────────────────────────────
with ec:
    st.markdown('<div class="sh">LIVE EQUATIONS (§7.3–7.4)</div>',
                unsafe_allow_html=True)
    atk_rows = [r for r in rows if r["is_atk"]]

    if atk_rows:
        r0  = atk_rows[0]
        f0  = r0["f"]
        b0  = BL.get(r0["cls"], BL["Normal"])
        ex  = "icmp_packets"
        zj  = abs(f0[ex] - b0[ex][0]) / (b0[ex][1] + EPS)
        risk = sum(W[k] * min(abs(f0[k] - PMU[k]) / (PSIG[k] + EPS), Z_MAX)
                   for k in f0)
        csa_v = round(100. * np.exp(-risk), 1)
        ev_c  = r0["gates"].get("ewma",  (False, 0))[1]
        cv_c  = r0["gates"].get("corr",  (False, 0))[1]
        se_c  = r0["gates"].get("stride",(False, 0))[1]

        st.markdown(f"""
<div class="eqp"><div class="eqt">Eq 7.2 — EWMA Z-score</div>
<div class="eqb">Z_i = |f_i(x') − μ_i| / σ_i<br>
{ex}:<br>Z = <span class="{'eqe' if zj>TAU_Z else 'eqv'}">{zj:.2f}</span>
(τ = {TAU_Z})</div></div>

<div class="eqp"><div class="eqt">Eq 7.3 — T_EWMA gate</div>
<div class="eqb">Fail if violations ≥ N/2<br>
count = <span class="{'eqe' if ev_c>=4 else 'eqw' if ev_c>=2 else 'eqv'}">{ev_c}</span> / 7</div></div>

<div class="eqp"><div class="eqt">Eq 7.5 — C_corr gate</div>
<div class="eqb">Fail if causal violations ≥ 3<br>
count = <span class="{'eqe' if cv_c>=3 else 'eqv'}">{cv_c}</span></div></div>

<div class="eqp"><div class="eqt">Eq 7.6 — S_STRIDE gate</div>
<div class="eqb">Fail if E = 0<br>
(all signature feats suppressed)<br>
elevated = <span class="{'eqe' if se_c==0 else 'eqv'}">{se_c}</span></div></div>

<div class="eqp"><div class="eqt">Eq 7.12 — Posture score</div>
<div class="eqb">Risk = {risk:.3f}<br>
CSA = 100·e^(−{risk:.3f})<br>
    = <span class="{'eqe' if csa_v<40 else 'eqw' if csa_v<75 else 'eqv'}">{csa_v}</span>%</div></div>
""", unsafe_allow_html=True)

    else:
        r0   = rows[0]
        f0   = r0["f"]
        risk = sum(W[k] * min(abs(f0[k] - PMU[k]) / (PSIG[k] + EPS), Z_MAX)
                   for k in f0)
        csa_v = round(100. * np.exp(-risk), 1)
        st.markdown(f"""
<div class="eqp"><div class="eqt">Eq 7.12 — Normal posture</div>
<div class="eqb">Benign traffic:<br>
z_j ≈ 0.03–0.07 (very small)<br>
Risk = {risk:.4f} ≈ 0<br>
CSA = <span class="eqv">{csa_v}</span>% ≈ 100 ✓</div></div>

<div class="eqp"><div class="eqt">Eq 7.8 — Master gate</div>
<div class="eqb">Block = (Φ ∧ IF_src)<br>
     ∨ (Φ ∧ IF_norm)<br>
     ∨ (IF_src ∧ IF_norm)<br>
Φ = T_EWMA ∨ C_corr ∨ S_STRIDE<br>
All gates: <span class="eqv">PASS ✓</span></div></div>

<div class="eqp"><div class="eqt">Eq 7.11 — Risk(t)</div>
<div class="eqb">Risk = Σ wⱼ·min(zⱼ,z_max)<br>
Weights w_j:<br>
cpu=.20  mem=.15  load=.15<br>
tcp=.15  icmp=.15  lat=.10</div></div>
""", unsafe_allow_html=True)

# ── STRIDE Radar ──────────────────────────────────────────────────────
with rc:
    st.markdown('<div class="sh">STRIDE RADAR (§6.3)</div>',
                unsafe_allow_html=True)
    sc_d = {"DOS": 0, "Spoofing": 0, "Tampering": 0, "Info_Disclosure": 0}
    for r in rows:
        if r["is_atk"] and r["cls"] in sc_d:
            sc_d[r["cls"]] += 1
    cats = list(sc_d.keys())
    vn   = [(sc_d[c] / (max(sc_d.values()) + 1)) * 100 for c in cats]
    fig_r = go.Figure(go.Scatterpolar(
        r=vn + [vn[0]],
        theta=[c.replace("_", " ") for c in cats] + [cats[0].replace("_", " ")],
        fill="toself",
        fillcolor="rgba(119,34,187,0.10)",
        line=dict(color="#7722bb", width=1.5),
        marker=dict(color="#7722bb", size=4),
    ))
    fig_r.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,100],
                            gridcolor="#1f344a", # BRIGHTENED
                            tickfont=dict(color="#8aaccc", size=8, # BRIGHTENED
                                          family="JetBrains Mono"),
                            linecolor="#1f344a"), # BRIGHTENED
            angularaxis=dict(gridcolor="#1f344a", # BRIGHTENED
                             tickfont=dict(color="#8aaccc", size=9, # BRIGHTENED
                                           family="JetBrains Mono"),
                             linecolor="#1f344a"), # BRIGHTENED
        ),
        height=280, margin=dict(l=15, r=15, t=15, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8aaccc", family="JetBrains Mono"),
        showlegend=False,
    )
    st.plotly_chart(fig_r, use_container_width=True,
                    key=f"r_{st.session_state.tick}")


# ─────────────────────────────────────────────────────────────────────
# ROW 4 — Z-SCORE HEATMAP  (Phase 1 / Phase 2 output)
# ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="sh">PHASE 1 OUTPUT — FEATURE Z-SCORE HEATMAP  '
    'z_j = |F_j − μ_j| / σ_j  (Eq 5.2)</div>',
    unsafe_allow_html=True)

fk  = ["cpu","memory","disk","load","tcp_conn","icmp_packets","latency"]
bn  = BL["Normal"]
zm  = [];  dl = []
for r in rows:
    zm.append([min(abs(r["f"][k] - bn[k][0]) / (bn[k][1] + EPS), 8.)
               for k in fk])
    dl.append(r["dev"] + (" ⚠" if r["is_atk"] else ""))
za = np.array(zm)

fig_h = go.Figure(go.Heatmap(
    z=za, x=fk, y=dl,
    colorscale=[
        [0.,   "#030a12"],
        [0.18, "#050e1c"],
        [0.45, "#091a09"],
        [0.70, "#181e00"],
        [1.,   "#2a0000"],
    ],
    zmin=0, zmax=7,
    colorbar=dict(
        title=dict(text="z",
                   font=dict(color="#8aaccc", size=9, # BRIGHTENED
                             family="JetBrains Mono")),
        tickfont=dict(color="#8aaccc", size=8, family="JetBrains Mono"), # BRIGHTENED
        bgcolor="rgba(0,0,0,0)",
        bordercolor="#081622",
    ),
    text=[[f"{v:.1f}" for v in row] for row in za],
    texttemplate="%{text}",
    textfont=dict(size=8, color="#e0e6ed", family="JetBrains Mono"), # BRIGHTENED to solid bright white/blue
))
fig_h.update_layout(
    height=230,
    margin=dict(l=0, r=0, t=5, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono", color="#8aaccc", size=9),
    xaxis=dict(tickfont=dict(color="#8aaccc", family="JetBrains Mono"), # BRIGHTENED
               gridcolor="#071018"),
    yaxis=dict(tickfont=dict(color="#8aaccc", family="JetBrains Mono"), # BRIGHTENED
               gridcolor="#071018"),
)
st.plotly_chart(fig_h, use_container_width=True,
                key=f"h_{st.session_state.tick}")


# ─────────────────────────────────────────────────────────────────────
# ROW 5 — EVENT LOG
# ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="sh">SECURITY EVENT LOG — '
    'PHASE 4 GATE DECISIONS (§7.3)</div>',
    unsafe_allow_html=True)

evs = list(st.session_state.event_log)[:25]
if evs:
    lh = ""
    for ev in evs:
        tc_  = "#b366ff" if ev["tag"] == "ADV" else "#ff4444"
        mc_  = "#ff4444" if ev["ml_miss"]  else "#00e676"
        cc_  = "#00e676" if ev["blocked"]  else "#ff4444"
        rstr = "+".join(ev["reasons"]) if ev["reasons"] else "—"
        cond = f" [Cond.{ev['cond']}]" if ev.get("cond") else ""
        lh += (
            f'<div style="border-bottom:1px solid #1a2a3a;padding:2px 0">'
            f'<span style="color:#66b3ff">[T{ev["t"]:04d}]</span> '
            f'<span style="color:{tc_};font-weight:600;">{ev["tag"]}</span> '
            f'<span style="color:#ffcc00">{ev["dev"]}</span> '
            f'<span style="color:#00e5ff">{ev["cls"]}</span> '
            f'ML:<span style="color:{mc_}">{"MISSED" if ev["ml_miss"] else "CAUGHT"}</span> '
            f'CSA:<span style="color:{cc_}">'
            f'{"BLOCKED("+rstr+")"+cond if ev["blocked"] else "EVADED"}'
            f'</span></div>'
        )
    st.markdown(f'<div class="elog">{lh}</div>', unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="elog" style="color:#66b3ff">'
        'No events yet — press RUN then select an attack scenario.'
        '</div>',
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="foot">
CSA Framework · Shahbaz Ali Imran · Birmingham City University · March 2026 ·
Phase 1 §4.4 Eq 4.1–4.3 · Phase 2 §5.3 Eq 5.2–5.4 · Phase 3 §6.4 RF F1=96.7% ·
Phase 4 §7.3 Eq 7.2–7.9 · Phase 5 §7.4 Eq 7.10–7.12 · CSA(t)=100·e^(−Risk(t))
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# AUTO-REFRESH
# ─────────────────────────────────────────────────────────────────────
if st.session_state.running:
    time.sleep(1.8)
    st.rerun()
