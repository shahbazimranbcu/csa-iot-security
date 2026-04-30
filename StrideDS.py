"""
==========================================================================
 CSA ADVERSARIAL ROBUSTNESS — SMARTHOME DATASET
 HopSkipJump (label-only) + CSA Verification Layer
 N_TRIALS=5 with 95% CI | Live ASR per sample | Oblivious + Adaptive
==========================================================================
MODEL: random_forest_model_normal.pkl
To switch to adversarial model change MODEL_FILE to:
  MODEL_FILE = "random_forest_model_adver.pkl"
==========================================================================
"""

import csv
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from art.estimators.classification import BlackBoxClassifier
from art.attacks.evasion import HopSkipJump

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ================================================================
# >>> SWAP THIS LINE TO SWITCH NORMAL / ADVERSARIAL
# ================================================================
MODEL_FILE     = "random_forest_model_adver.pkl"
# MODEL_FILE   = "random_forest_model_normal.pkl"

TRAIN_AUG_FILE = "train_augmented.csv"
TEST_FILE      = "test.csv"
LABEL_COL      = "target"
OUTPUT_DIR     = Path("results_smarthome")
OUTPUT_DIR.mkdir(exist_ok=True)

# ================================================================
# TUNING KNOBS
# ================================================================
RANDOM_SEED     = 42
MAX_SAMPLES     = 100
N_TRIALS        = 5
HSJ_MAX_ITER    = 10
ZSCORE_THRESH   = 1.8
ZSCORE_MAJORITY = 0.1
ISO_CONTAM      = 0.12
STRIDE_MULT     = 1.03
EWMA_ALPHA      = 0.3
SCALING_FACTORS = [0.01, 0.05, 0.10, 0.15]
W_EWMA          = 2.0
W_CORR          = 3.0
W_STRIDE        = 2.0

# ================================================================
# FEATURE CONFIG
# ================================================================
ALL_FEATURES = [
    "cpu", "memory", "disk", "load", "uptime",
    "bytes_sent", "bytes_recv", "tcp_conn",
    "icmp_packets", "open_ports_count", "latency",
]
SELECTED_FEATURES = [
    "cpu", "memory", "disk", "load",
    "tcp_conn", "icmp_packets", "latency",
]
FEAT_BOUNDS = {
    "cpu"         : (0.0,    100.0),
    "memory"      : (5.9963,  64.4606),
    "disk"        : (68.0,    95.0),
    "load"        : (0.0,     28.99),
    "tcp_conn"    : (2.0,     45.0),
    "icmp_packets": (2.0, 134562.0),
    "latency"     : (0.0,   6182.53),
}
CORRELATION_RULES = [
    ("latency",      "tcp_conn",   0.05,  -1),
    ("cpu",          "load",       0.10,  +1),
    ("icmp_packets", "latency",  0.0001,  +1),
    ("bytes_sent",   "bytes_recv",  0.0,  +1),
]
CORRELATION_THRESHOLD = 3

STRIDE_SIGNATURE = {
    "DOS":             ["icmp_packets", "tcp_conn"],
    "Spoofing":        ["icmp_packets", "latency"],
    "Tampering":       ["cpu", "memory", "load"],
    "Info_Disclosure": ["cpu", "disk"],
}
CLASS_NAMES     = ["Normal", "DOS", "Spoofing", "Tampering", "Info_Disclosure"]
NORMAL_IDX      = CLASS_NAMES.index("Normal")
EXECUTION_ORDER = ["Tampering", "Spoofing", "DOS", "Info_Disclosure", "Normal"]
IF_CONTAMINATION = {
    "Normal": 0.05, "DOS": 0.50,
    "Spoofing": 0.50, "Tampering": 0.50,
    "Info_Disclosure": 0.50,
}
SELECTED_IDX = [ALL_FEATURES.index(f) for f in SELECTED_FEATURES]


# ================================================================
# DATA LOADING
# ================================================================
def load_data():
    log.info("Loading train_augmented.csv for CSA baselines...")
    df_aug = pd.read_csv(TRAIN_AUG_FILE)
    df_aug = df_aug.apply(pd.to_numeric, errors="coerce").dropna()
    X_aug  = df_aug[ALL_FEATURES].values.astype(np.float32)
    y_aug  = df_aug[LABEL_COL].values.astype(int)

    log.info("Loading test.csv, splitting 50%% val / 50%% eval...")
    df_test = pd.read_csv(TEST_FILE)
    df_test = df_test.apply(pd.to_numeric, errors="coerce").dropna()
    X_test  = df_test[ALL_FEATURES].values.astype(np.float32)
    y_test  = df_test[LABEL_COL].values.astype(int)

    X_val, X_eval, y_val, y_eval = train_test_split(
        X_test, y_test, test_size=0.5,
        stratify=y_test, random_state=RANDOM_SEED,
    )
    log.info("Val=%d  Eval=%d", len(X_val), len(X_eval))
    return X_aug, y_aug, X_val, y_val, X_eval, y_eval


# ================================================================
# CSA BASELINES
# ================================================================
def fit_baselines(X_aug, y_aug):
    baselines = {}
    for idx, name in enumerate(CLASS_NAMES):
        mask  = y_aug == idx
        X_cls = X_aug[mask][:, SELECTED_IDX]
        if len(X_cls) < 5:
            continue
        bl = {}
        for j, fname in enumerate(SELECTED_FEATURES):
            series    = pd.Series(X_cls[:, j])
            ewma_vals = series.ewm(alpha=EWMA_ALPHA, adjust=False).mean().values
            bl[fname] = {
                "ewma_mean": float(np.mean(ewma_vals)),
                "ewma_std":  float(np.std(ewma_vals))  + 1e-9,
                "raw_mean":  float(np.mean(X_cls[:, j])),
                "raw_std":   float(np.std(X_cls[:, j])) + 1e-9,
            }
        baselines[idx] = bl
    return baselines


def calibrate_zscore(X_val, y_val, baselines):
    X_normal = X_val[y_val == NORMAL_IDX]
    best_z, best_fpr = 3.0, 1.0
    for z in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]:
        flagged = 0
        for s in X_normal:
            viols = sum(
                1 for j, fname in enumerate(SELECTED_FEATURES)
                if abs(s[SELECTED_IDX[j]] - baselines[NORMAL_IDX][fname]["ewma_mean"])
                   / baselines[NORMAL_IDX][fname]["ewma_std"] > z
            )
            if viols >= len(SELECTED_FEATURES) // 2:
                flagged += 1
        fpr = flagged / max(len(X_normal), 1)
        log.info("  z=%.1f  FPR=%.2f%%", z, fpr * 100)
        if fpr <= 0.05 and best_fpr > 0.05:
            best_z, best_fpr = z, fpr
    log.info("Selected z=%.1f (FPR=%.2f%%)", best_z, best_fpr * 100)
    return best_z


def fit_iso_forests(X_aug, y_aug):
    iso = {}
    for idx, name in enumerate(CLASS_NAMES):
        mask  = y_aug == idx
        X_cls = X_aug[mask][:, SELECTED_IDX]
        if len(X_cls) < 10:
            X_cls = X_aug[:, SELECTED_IDX]
        cont = IF_CONTAMINATION.get(name, 0.05)
        iso[idx] = IsolationForest(
            n_estimators=200, contamination=cont,
            random_state=RANDOM_SEED, n_jobs=-1,
        ).fit(X_cls)
        sf_rate = np.mean(iso[idx].predict(X_cls) == -1) * 100
        log.info("  IF[%s]  n=%d  contam=%.2f  self-flag=%.1f%%",
                 name, len(X_cls), cont, sf_rate)
    return iso


# ================================================================
# CSA RULES
# ================================================================
def rule_tmva(x_adv, cls_idx, baselines, z_thresh):
    bl = baselines.get(cls_idx)
    if bl is None:
        return True
    viols = sum(
        1 for j, fname in enumerate(SELECTED_FEATURES)
        if abs(x_adv[SELECTED_IDX[j]] - bl[fname]["ewma_mean"])
           / bl[fname]["ewma_std"] > z_thresh
    )
    return (viols / len(SELECTED_FEATURES)) < ZSCORE_MAJORITY


def rule_ccorr(x_orig, x_adv):
    feat_map   = {f: i for i, f in enumerate(SELECTED_FEATURES)}
    violations = 0
    for (fA, fB, min_ratio, expected_sign) in CORRELATION_RULES:
        if fA not in feat_map or fB not in feat_map:
            continue
        iA = SELECTED_IDX[feat_map[fA]]
        iB = SELECTED_IDX[feat_map[fB]]
        dA = float(x_adv[iA]) - float(x_orig[iA])
        dB = float(x_adv[iB]) - float(x_orig[iB])
        rng_A = FEAT_BOUNDS[fA][1] - FEAT_BOUNDS[fA][0]
        if abs(dA) > 0.01 * rng_A:
            actual_sign = 1 if dA * dB > 0 else -1
            if expected_sign != 0 and actual_sign != expected_sign:
                violations += 1
            elif expected_sign == 0:
                if (dA > 0 and dB < min_ratio * dA) or \
                   (dA < 0 and dB > min_ratio * dA):
                    violations += 1
    return violations < CORRELATION_THRESHOLD


def rule_sstride(x_adv, cls_idx, baselines):
    """Fires when NO signature feature is elevated — neutered-attack paradox."""
    cls_name  = CLASS_NAMES[cls_idx]
    if cls_idx == NORMAL_IDX:
        return True
    sig_feats = STRIDE_SIGNATURE.get(cls_name)
    if not sig_feats:
        return True
    bl = baselines.get(cls_idx)
    if bl is None:
        return True
    feat_map = {f: i for i, f in enumerate(SELECTED_FEATURES)}
    for fname in sig_feats:
        if fname not in feat_map:
            continue
        threshold = bl[fname]["raw_mean"] * STRIDE_MULT
        if float(x_adv[SELECTED_IDX[feat_map[fname]]]) >= threshold:
            return True   # at least one elevated → passes
    return False          # none elevated → flag


def csa_detect(x_orig, x_adv, cls_idx, iso, baselines, z_thresh, mode="FULL"):
    tmva_ok   = rule_tmva(x_adv, cls_idx, baselines, z_thresh)  if mode != "NO_TMVA"   else True
    corr_ok   = rule_ccorr(x_orig, x_adv)                       if mode != "NO_CORR"   else True
    stride_ok = rule_sstride(x_adv, cls_idx, baselines)          if mode != "NO_STRIDE" else True
    violations = sum([not tmva_ok, not corr_ok, not stride_ok])

    x_sel = x_adv[SELECTED_IDX].reshape(1, -1)
    if mode != "NO_IF":
        if_flagged = (iso[cls_idx].predict(x_sel)[0] == -1 and
                      iso[NORMAL_IDX].predict(x_sel)[0] == -1)
    else:
        if_flagged = False

    detected = (not corr_ok) or (violations >= 2) or if_flagged
    reasons  = []
    if not tmva_ok:   reasons.append("TMVA")
    if not corr_ok:   reasons.append("CORR")
    if not stride_ok: reasons.append("STRIDE")
    if mode != "NO_IF" and iso[cls_idx].predict(x_sel)[0] == -1:
        reasons.append("SRC_IF")
    if mode != "NO_IF" and iso[NORMAL_IDX].predict(x_sel)[0] == -1:
        reasons.append("NORM_IF")
    return {"detected": detected, "reason": "+".join(reasons) if detected else "NONE"}


# ================================================================
# ART BLACK-BOX
# ================================================================
def make_blackbox(model):
    def predict_fn(x_np):
        df      = pd.DataFrame(x_np.astype(np.float32), columns=ALL_FEATURES)
        preds   = model.predict(df)
        one_hot = np.zeros((len(preds), len(CLASS_NAMES)), dtype=np.float32)
        for i, p in enumerate(preds):
            one_hot[i, int(p)] = 1.0
        return one_hot
    return BlackBoxClassifier(
        predict_fn,
        input_shape=(len(ALL_FEATURES),),
        nb_classes=len(CLASS_NAMES),
        clip_values=(0.0, 1e6),
    )


# ================================================================
# ADAPTIVE LOSS
# ================================================================
def adaptive_loss(x_adv, x_orig, cls_idx, model, baselines, z_thresh):
    proba   = model.predict_proba(x_adv.reshape(1, -1))[0]
    ml_loss = 1.0 - proba[cls_idx]

    bl = baselines.get(cls_idx, {})
    viols_tmva = sum(
        1 for j, fname in enumerate(SELECTED_FEATURES)
        if abs(x_adv[SELECTED_IDX[j]] - bl.get(fname, {}).get("ewma_mean", 0))
           / bl.get(fname, {}).get("ewma_std", 1) > z_thresh
    ) / len(SELECTED_FEATURES)

    feat_map = {f: i for i, f in enumerate(SELECTED_FEATURES)}
    cv = 0
    for (fA, fB, min_ratio, expected_sign) in CORRELATION_RULES:
        if fA not in feat_map or fB not in feat_map:
            continue
        iA = SELECTED_IDX[feat_map[fA]]
        iB = SELECTED_IDX[feat_map[fB]]
        dA = float(x_adv[iA]) - float(x_orig[iA])
        dB = float(x_adv[iB]) - float(x_orig[iB])
        rA = FEAT_BOUNDS[fA][1] - FEAT_BOUNDS[fA][0]
        if abs(dA) > 0.01 * rA:
            actual_sign = 1 if dA * dB > 0 else -1
            if expected_sign != 0 and actual_sign != expected_sign:
                cv += 1
    viols_corr = cv / max(len(CORRELATION_RULES), 1)

    cls_name  = CLASS_NAMES[cls_idx]
    sig_feats = STRIDE_SIGNATURE.get(cls_name, [])
    elevated  = sum(
        1 for fname in sig_feats
        if fname in feat_map
        and x_adv[SELECTED_IDX[feat_map[fname]]]
           >= bl.get(fname, {}).get("raw_mean", 0) * STRIDE_MULT
    )
    viols_stride = 1.0 if (elevated == 0 and len(sig_feats) > 0) else 0.0

    return ml_loss - W_EWMA * viols_tmva - W_CORR * viols_corr - W_STRIDE * viols_stride


def _adaptive_hill_climb(x, cls_idx, model, baselines, z_thresh, epsilons, rng):
    current      = x.copy()
    current_loss = adaptive_loss(current, x, cls_idx, model, baselines, z_thresh)
    best_adv, best_loss = current.copy(), current_loss

    for iteration in range(300):
        candidate = current.copy()
        for i, fname in zip(SELECTED_IDX, SELECTED_FEATURES):
            eps = epsilons[fname]
            lo  = max(FEAT_BOUNDS[fname][0], x[i] - eps)
            hi  = min(FEAT_BOUNDS[fname][1], x[i] + eps)
            candidate[i] = float(np.clip(current[i] + rng.uniform(-eps, eps), lo, hi))

        cand_loss = adaptive_loss(candidate, x, cls_idx, model, baselines, z_thresh)
        df_c      = pd.DataFrame(candidate.reshape(1, -1), columns=ALL_FEATURES)
        # FIX: early exit consistent with paper metric direction
        _pred = int(model.predict(df_c)[0])
        _ok   = (_pred != NORMAL_IDX) if cls_idx == NORMAL_IDX else (_pred == NORMAL_IDX)
        if _ok:
            return candidate, iteration + 1

        if cand_loss > current_loss:
            current      = candidate.copy()
            current_loss = cand_loss
            if cand_loss > best_loss:
                best_loss = cand_loss
                best_adv  = candidate.copy()

    return best_adv, 300


# ================================================================
# CI HELPER
# ================================================================
def mean_ci(values, confidence=0.95):
    n = len(values)
    m = float(np.mean(values))
    if n < 2:
        return m, 0.0
    h = stats.sem(values) * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return m, float(h)


# ================================================================
# SINGLE TRIAL
# ================================================================
def run_trial(
    attack_hsj, model, X_eval, y_eval,
    iso, baselines, z_thresh,
    sf, rng, mode, attack_mode,
    trial_num, out_csv, csv_columns,
):
    epsilons = {f: sf * (FEAT_BOUNDS[f][1] - FEAT_BOUNDS[f][0])
                for f in SELECTED_FEATURES}

    total = fooled = csa_pass = errors = 0

    print(f"\n  {'─'*100}")
    print(f"  Trial {trial_num} | mode={mode} | sf={sf:.2f} | attack={attack_mode}")
    print(f"  {'─'*100}")
    print(f"  {'#':>5}  {'Class':>16}  {'PredAfter':>16}  {'Result':>9}  "
          f"{'Iters':>6}  {'ML-ASR':>7}  {'CSA-ASR':>8}  {'CSA':>6}  {'Reason':<22}")
    print(f"  {'─'*5}  {'─'*16}  {'─'*16}  {'─'*9}  {'─'*6}  "
          f"{'─'*7}  {'─'*8}  {'─'*6}  {'─'*22}")

    for cls_name in EXECUTION_ORDER:
        cls_idx = CLASS_NAMES.index(cls_name)
        idx     = np.where(y_eval == cls_idx)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=min(MAX_SAMPLES, len(idx)), replace=False)

        for x_raw in X_eval[chosen]:
            x = x_raw.astype(np.float32)

            try:
                if attack_mode == "oblivious":
                    x_adv   = attack_hsj.generate(x=x.reshape(1, -1))[0].astype(np.float32)
                    n_iters = HSJ_MAX_ITER
                else:
                    x_adv, n_iters = _adaptive_hill_climb(
                        x, cls_idx, model, baselines, z_thresh, epsilons, rng
                    )
            except Exception as e:
                errors += 1
                log.debug("Attack error: %s", e)
                continue

            # Clip to epsilon budget
            for j, fname in enumerate(SELECTED_FEATURES):
                lo = max(FEAT_BOUNDS[fname][0], x[SELECTED_IDX[j]] - epsilons[fname])
                hi = min(FEAT_BOUNDS[fname][1], x[SELECTED_IDX[j]] + epsilons[fname])
                x_adv[SELECTED_IDX[j]] = float(np.clip(x_adv[SELECTED_IDX[j]], lo, hi))

            df_orig = pd.DataFrame(x.reshape(1, -1),     columns=ALL_FEATURES)
            df_adv  = pd.DataFrame(x_adv.reshape(1, -1), columns=ALL_FEATURES)
            p_orig  = int(model.predict(df_orig)[0])
            p_adv   = int(model.predict(df_adv)[0])

            total += 1
            # ── PAPER METRIC: attack→Normal = success ──────────────────────
            if cls_idx == NORMAL_IDX:
                success = (p_adv != NORMAL_IDX)   # Normal pushed toward attack
            else:
                success = (p_adv == NORMAL_IDX)   # Attack hidden as Normal
            # ───────────────────────────────────────────────────────────────

            csa_result  = {"detected": False, "reason": "N/A"}
            intercepted = False
            if success:
                fooled     += 1
                csa_result  = csa_detect(x, x_adv, cls_idx, iso, baselines, z_thresh, mode)
                intercepted = csa_result["detected"]
                if not intercepted:
                    csa_pass += 1

            live_ml    = fooled   / total * 100
            live_csa   = csa_pass / total * 100
            result_str = "SUCCESS ✓" if success else "FAIL    ✗"
            csa_str    = "CATCH" if intercepted else ("PASS" if success else "─")

            print(
                f"  {total:>5}  {cls_name:>16}  {CLASS_NAMES[p_adv]:>16}  "
                f"{result_str:>9}  {n_iters:>6}  {live_ml:>6.1f}%  "
                f"{live_csa:>7.1f}%  {csa_str:>6}  {csa_result['reason']:<22}"
            )

            with open(out_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
                w.writerow({
                    "trial":           trial_num,
                    "mode":            mode,
                    "attack_mode":     attack_mode,
                    "scaling_factor":  sf,
                    "class_true":      cls_name,
                    "success":         int(success),
                    "csa_detected":    int(intercepted),
                    "csa_net_success": int(success and not intercepted),
                    "csa_reason":      csa_result["reason"],
                    "iterations_used": n_iters,
                    "pred_before":     CLASS_NAMES[p_orig],
                    "pred_after":      CLASS_NAMES[p_adv],
                    "live_ml_asr":     round(live_ml,  2),
                    "live_csa_asr":    round(live_csa, 2),
                })

    denom   = max(total, 1)
    ml_asr  = fooled   / denom * 100
    csa_asr = csa_pass / denom * 100
    print(f"\n  TRIAL {trial_num} SUMMARY | sf={sf:.2f} | {mode} | {attack_mode}")
    print(f"    Total={total}  Fooled={fooled}  CSA-pass={csa_pass}  Errors={errors}")
    print(f"    ML-ASR={ml_asr:.2f}%  CSA-ASR={csa_asr:.2f}%  "
          f"Reduction={ml_asr - csa_asr:.2f}pp  "
          f"Joint={ml_asr * csa_asr / 100:.2f}%")
    return ml_asr, csa_asr, total, fooled, csa_pass


def _write_row(path, columns, data):
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=columns, extrasaction="ignore").writerow(data)


# ================================================================
# MAIN
# ================================================================
def main():
    model_name = Path(MODEL_FILE).stem
    log.info("Model: %s", model_name)

    model = joblib.load(MODEL_FILE)
    log.info("Trees: %d", len(model.estimators_))

    X_aug, y_aug, X_val, y_val, X_eval, y_eval = load_data()

    log.info("Fitting baselines...")
    baselines = fit_baselines(X_aug, y_aug)

    log.info("Calibrating z-score threshold...")
    z_thresh = calibrate_zscore(X_val, y_val, baselines)

    log.info("Fitting Isolation Forests...")
    iso = fit_iso_forests(X_aug, y_aug)

    log.info("Building label-only BlackBoxClassifier for HSJ...")
    bb_clf     = make_blackbox(model)
    attack_hsj = HopSkipJump(
        classifier=bb_clf,
        max_iter=HSJ_MAX_ITER,
        targeted=False,
        verbose=False,
    )

    X_eval_normal = X_eval[y_eval == NORMAL_IDX]
    fp  = sum(1 for s in X_eval_normal
              if csa_detect(s, s, NORMAL_IDX, iso, baselines, z_thresh)["detected"])
    fpr = fp / max(len(X_eval_normal), 1) * 100
    log.info("CSA FPR on clean Normal (eval): %.2f%%", fpr)

    ABLATION_MODES = ["FULL", "NO_TMVA", "NO_CORR", "NO_STRIDE", "NO_IF"]
    ATTACK_MODES   = ["oblivious", "adaptive"]

    csv_columns = [
        "trial", "mode", "attack_mode", "scaling_factor", "class_true",
        "success", "csa_detected", "csa_net_success", "csa_reason",
        "iterations_used", "pred_before", "pred_after",
        "live_ml_asr", "live_csa_asr",
    ]
    out_csv = OUTPUT_DIR / f"{model_name}_results.csv"
    with open(out_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_columns).writeheader()

    all_records = []

    for attack_mode in ATTACK_MODES:
        sfs = SCALING_FACTORS if attack_mode == "oblivious" else [0.15]
        for sf in sfs:
            for mode in ABLATION_MODES:
                log.info("═══ attack=%s | sf=%.2f | mode=%s ═══",
                         attack_mode, sf, mode)
                ml_list, csa_list = [], []
                for trial in range(1, N_TRIALS + 1):
                    rng = np.random.default_rng(RANDOM_SEED + trial)
                    ml_asr, csa_asr, *_ = run_trial(
                        attack_hsj, model, X_eval, y_eval,
                        iso, baselines, z_thresh,
                        sf, rng, mode, attack_mode,
                        trial, out_csv, csv_columns,
                    )
                    ml_list.append(ml_asr)
                    csa_list.append(csa_asr)

                m_ml,  h_ml  = mean_ci(ml_list)
                m_csa, h_csa = mean_ci(csa_list)
                joint = m_ml * m_csa / 100.0
                all_records.append({
                    "model":       model_name,
                    "attack_mode": attack_mode,
                    "sf":          sf,
                    "mode":        mode,
                    "ML-ASR":      f"{m_ml:.2f} ± {h_ml:.2f}",
                    "CSA-ASR":     f"{m_csa:.2f} ± {h_csa:.2f}",
                    "Reduction":   round(m_ml - m_csa, 2),
                    "Joint_%":     round(joint, 2),
                })
                log.info("RESULT: ML=%.2f±%.2f  CSA=%.2f±%.2f  "
                         "Red=%.2fpp  Joint=%.2f%%",
                         m_ml, h_ml, m_csa, h_csa, m_ml - m_csa, joint)

    df_res = pd.DataFrame(all_records)
    print("\n" + "=" * 90)
    print(f"  FINAL SUMMARY — {model_name}")
    print("=" * 90)
    print(df_res.to_string(index=False))

    summary_csv = OUTPUT_DIR / f"{model_name}_summary.csv"
    df_res.to_csv(summary_csv, index=False)

    full_obl = df_res[
        (df_res["mode"] == "FULL") & (df_res["attack_mode"] == "oblivious")
    ]
    rows_tex = "\n".join(
        f"  {r['sf']:.2f} & {r['ML-ASR']:<18} & {r['CSA-ASR']:<18} "
        f"& {r['Reduction']:>6} & {r['Joint_%']:>6} \\\\"
        for _, r in full_obl.iterrows()
    )
    latex = (
        "\\begin{table}[t]\\centering\n"
        f"\\caption{{SmartHome CSA — {model_name} ({N_TRIALS} trials, 95\\% CI)}}\n"
        "\\label{tab:smarthome_results}\n"
        "\\begin{tabular}{lcccc}\\toprule\n"
        "$\\alpha$ & ML-ASR (\\%) & CSA-ASR (\\%) & Red (pp) & Joint (\\%) \\\\\n"
        "\\midrule\n"
        f"{rows_tex}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}"
    )
    (OUTPUT_DIR / f"{model_name}_table.tex").write_text(latex)
    log.info("Results → %s", summary_csv)
    log.info("CSA FPR on clean Normal: %.2f%%", fpr)


if __name__ == "__main__":
    main()
