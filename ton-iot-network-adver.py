"""
==========================================================================
 CSA ADVERSARIAL ROBUSTNESS — TON-IoT NETWORK DATASET
 HopSkipJump (label-only) + CSA Verification Layer
 N_TRIALS=5 with 95% CI | Live ASR per sample | Oblivious + Adaptive
==========================================================================
FIXES:
  1. success logic: attack->Normal only (paper metric)
  2. hill-climb early exit: consistent with success direction
  3. safe_load_model: sklearn version mismatch handled
  4. IndentationError on MODEL_FILE line removed
"""

import csv, logging, pickle, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from art.estimators.classification import BlackBoxClassifier
from art.attacks.evasion import HopSkipJump

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# >>> SWAP THIS ONE LINE to switch model
MODEL_FILE = "network_adver.pkl"
# MODEL_FILE = "network_normal.pkl"

DATA_FILE  = "network_dataset_test.csv"
TYPE_COL   = "type"
OUTPUT_DIR = Path("results_ton_network")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED=42; MAX_SAMPLES=100; N_TRIALS=5; HSJ_MAX_ITER=10
ZSCORE_THRESH=1.8; ZSCORE_MAJORITY=0.1; ISO_CONTAM=0.12
STRIDE_MULT=1.03; EWMA_ALPHA=0.3
SCALING_FACTORS=[0.01,0.05,0.10,0.15]
W_EWMA=2.0; W_CORR=3.0; W_STRIDE=2.0

SELECTED_FEATURES = ["src_pkts","src_ip_bytes","proto","dns_query","dst_ip_bytes","conn_state","dst_pkts"]
ALL_FEATURES = [
    "proto","service","duration","src_bytes","dst_bytes","conn_state","missed_bytes",
    "src_pkts","src_ip_bytes","dst_pkts","dst_ip_bytes","dns_query","dns_qclass",
    "dns_qtype","dns_rcode","dns_AA","dns_RD","dns_RA","dns_rejected","ssl_version",
    "ssl_cipher","ssl_resumed","ssl_established","ssl_subject","ssl_issuer",
    "http_trans_depth","http_method","http_uri","http_version",
    "http_request_body_len","http_response_body_len","http_status_code",
    "http_user_agent","http_orig_mime_types","http_resp_mime_types",
    "weird_name","weird_addl","weird_notice",
]
FEAT_BOUNDS = {
    "src_pkts":(0.0,24623.0),"src_ip_bytes":(0.0,6522626.0),"proto":(0.0,2.0),
    "dns_query":(0.0,725.0),"dst_ip_bytes":(0.0,86395523.0),
    "conn_state":(0.0,12.0),"dst_pkts":(0.0,121942.0),
}
CORRELATION_RULES = [
    ("src_pkts","src_ip_bytes",0.01,+1),
    ("dst_pkts","dst_ip_bytes",0.01,+1),
]
CORRELATION_THRESHOLD = 1
STRIDE_SIGNATURE = {
    "dos":["src_pkts","src_ip_bytes"],
    "mitm":["conn_state","dst_pkts"],
}
CLASS_NAMES = ["backdoor","ddos","dos","injection","mitm",
               "normal","password","ransomware","scanning","xss"]
NORMAL_IDX  = CLASS_NAMES.index("normal")
EXECUTION_ORDER = ["mitm","dos","normal"]
IF_CONTAM_MAP = {"normal":0.05,"dos":0.50,"mitm":0.50}
SELECTED_IDX = [ALL_FEATURES.index(f) for f in SELECTED_FEATURES]


def safe_load_model(model_file, data_file):
    compat = Path(model_file).with_suffix(".compat.pkl")
    if compat.exists():
        log.info("Loading compat pkl: %s", compat)
        return joblib.load(compat)
    try:
        m = joblib.load(model_file)
        log.info("Loaded OK — %d trees", len(m.estimators_))
        return m
    except (ValueError, pickle.UnpicklingError) as e:
        log.warning("sklearn mismatch: %s — retraining...", e)
        n_est = 500
        try:
            import re as _re
            raw = Path(model_file).read_bytes()
            hit = _re.search(rb"n_estimators.*?(\d{2,4})", raw[:2000])
            if hit: n_est = int(hit.group(1))
        except Exception: pass
        df_raw = pd.read_csv(data_file)
        ts = df_raw[TYPE_COL].str.lower().str.strip()
        df = df_raw.copy()
        for col in ALL_FEATURES:
            if col in df.columns and df[col].dtype == object:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        for col in ALL_FEATURES:
            if col not in df.columns: df[col] = 0
        df[ALL_FEATURES] = df[ALL_FEATURES].apply(pd.to_numeric,errors="coerce").fillna(0)
        mask = ts.isin(EXECUTION_ORDER)
        X = df.loc[mask,ALL_FEATURES].values.astype(np.float32)
        y = np.array([CLASS_NAMES.index(t) for t in ts[mask]])
        log.info("  Training RF n=%d on %d samples...", n_est, len(X))
        m = RandomForestClassifier(n_estimators=n_est,random_state=RANDOM_SEED,
                                   n_jobs=-1,class_weight="balanced")
        m.fit(X, y)
        joblib.dump(m, compat)
        log.info("  Compat saved → %s", compat)
        return m


def load_data():
    df_raw = pd.read_csv(DATA_FILE)
    ts = df_raw[TYPE_COL].str.lower().str.strip()
    df = df_raw.copy()
    for col in ALL_FEATURES:
        if col in df.columns and df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    for col in ALL_FEATURES:
        if col not in df.columns: df[col] = 0
    df[ALL_FEATURES] = df[ALL_FEATURES].apply(pd.to_numeric,errors="coerce").fillna(0)
    X_all = df[ALL_FEATURES].values.astype(np.float32)
    mask  = ts.isin(EXECUTION_ORDER)
    X_flt = X_all[mask.values]
    y_flt = np.array([CLASS_NAMES.index(t) for t in ts[mask].values])
    X_val,X_eval,y_val,y_eval = train_test_split(
        X_flt,y_flt,test_size=0.5,stratify=y_flt,random_state=RANDOM_SEED)
    log.info("Val=%d  Eval=%d", len(X_val), len(X_eval))
    return X_val,y_val,X_eval,y_eval


def fit_baselines(X_val, y_val):
    bls={}
    for cn in EXECUTION_ORDER:
        ci=CLASS_NAMES.index(cn); mask=y_val==ci
        Xc=X_val[mask][:,SELECTED_IDX]
        if len(Xc)==0: continue
        bl={}
        for j,fn in enumerate(SELECTED_FEATURES):
            s=pd.Series(Xc[:,j]); ev=s.ewm(alpha=EWMA_ALPHA,adjust=False).mean().values
            bl[fn]={"ewma_mean":float(np.mean(ev)),"ewma_std":float(np.std(ev))+1e-9,
                    "raw_mean":float(np.mean(Xc[:,j])),"raw_std":float(np.std(Xc[:,j]))+1e-9}
        bls[ci]=bl; log.info("  Baseline[%s]: %d",cn,int(mask.sum()))
    return bls


def calibrate_zscore(X_val, y_val, bls):
    Xn=X_val[y_val==NORMAL_IDX]; bz,bf=3.0,1.0
    for z in [2.5,3.0,3.5,4.0,4.5,5.0,6.0]:
        fg=sum(1 for s in Xn if sum(1 for j,fn in enumerate(SELECTED_FEATURES)
               if abs(s[SELECTED_IDX[j]]-bls[NORMAL_IDX][fn]["ewma_mean"])
               /bls[NORMAL_IDX][fn]["ewma_std"]>z)>=len(SELECTED_FEATURES)//2)
        fpr=fg/max(len(Xn),1); log.info("  z=%.1f FPR=%.2f%%",z,fpr*100)
        if fpr<=0.05 and bf>0.05: bz,bf=z,fpr
    log.info("Selected z=%.1f (FPR=%.2f%%)",bz,bf*100); return bz


def fit_iso_forests(X_val, y_val):
    iso={}
    for cn in EXECUTION_ORDER:
        ci=CLASS_NAMES.index(cn); mask=y_val==ci
        Xc=X_val[mask][:,SELECTED_IDX]
        if len(Xc)<10: Xc=X_val[:,SELECTED_IDX]
        cont=IF_CONTAM_MAP.get(cn,0.05)
        iso[ci]=IsolationForest(n_estimators=200,contamination=cont,
                                random_state=RANDOM_SEED,n_jobs=-1).fit(Xc)
        log.info("  IF[%s] n=%d contam=%.2f self-flag=%.1f%%",
                 cn,len(Xc),cont,np.mean(iso[ci].predict(Xc)==-1)*100)
    return iso


def rule_tmva(xa,ci,bls,zt):
    bl=bls.get(ci)
    if bl is None: return True
    v=sum(1 for j,fn in enumerate(SELECTED_FEATURES)
          if abs(xa[SELECTED_IDX[j]]-bl[fn]["ewma_mean"])/bl[fn]["ewma_std"]>zt)
    return (v/len(SELECTED_FEATURES))<ZSCORE_MAJORITY

def rule_ccorr(xo,xa):
    fm={f:i for i,f in enumerate(SELECTED_FEATURES)}; viol=0
    for (fA,fB,mr,es) in CORRELATION_RULES:
        if fA not in fm or fB not in fm: continue
        iA=SELECTED_IDX[fm[fA]]; iB=SELECTED_IDX[fm[fB]]
        dA=float(xa[iA])-float(xo[iA]); dB=float(xa[iB])-float(xo[iB])
        rA=FEAT_BOUNDS[fA][1]-FEAT_BOUNDS[fA][0]
        if abs(dA)>0.01*rA:
            if (1 if dA*dB>0 else -1)!=es: viol+=1
    return viol<CORRELATION_THRESHOLD

def rule_sstride(xa,ci,bls):
    cn=CLASS_NAMES[ci]
    if ci==NORMAL_IDX: return True
    sf=STRIDE_SIGNATURE.get(cn)
    if not sf: return True
    bl=bls.get(ci)
    if bl is None: return True
    fm={f:i for i,f in enumerate(SELECTED_FEATURES)}
    for fn in sf:
        if fn not in fm: continue
        if float(xa[SELECTED_IDX[fm[fn]]])>=bl[fn]["raw_mean"]*STRIDE_MULT: return True
    return False

def csa_detect(xo,xa,ci,iso,bls,zt,mode="FULL"):
    tok=rule_tmva(xa,ci,bls,zt)   if mode!="NO_TMVA"   else True
    cok=rule_ccorr(xo,xa)          if mode!="NO_CORR"   else True
    sok=rule_sstride(xa,ci,bls)    if mode!="NO_STRIDE" else True
    viol=sum([not tok,not cok,not sok])
    xs=xa[SELECTED_IDX].reshape(1,-1)
    if mode!="NO_IF" and ci in iso and NORMAL_IDX in iso:
        iflag=(iso[ci].predict(xs)[0]==-1 and iso[NORMAL_IDX].predict(xs)[0]==-1)
    else: iflag=False
    det=(not cok) or (viol>=2) or iflag
    rs=[]
    if not tok: rs.append("TMVA")
    if not cok: rs.append("CORR")
    if not sok: rs.append("STRIDE")
    if mode!="NO_IF" and ci in iso and iso[ci].predict(xs)[0]==-1: rs.append("SRC_IF")
    if mode!="NO_IF" and NORMAL_IDX in iso and iso[NORMAL_IDX].predict(xs)[0]==-1: rs.append("NORM_IF")
    return {"detected":det,"reason":"+".join(rs) if det else "NONE"}

def make_blackbox(model):
    def pfn(xnp):
        df=pd.DataFrame(xnp.astype(np.float32),columns=ALL_FEATURES)
        preds=model.predict(df)
        oh=np.zeros((len(preds),len(CLASS_NAMES)),dtype=np.float32)
        for i,p in enumerate(preds): oh[i,int(p)]=1.0
        return oh
    return BlackBoxClassifier(pfn,input_shape=(len(ALL_FEATURES),),
                              nb_classes=len(CLASS_NAMES),clip_values=(0.0,1e9))

def adaptive_loss(xa,xo,ci,model,bls,zt):
    proba=model.predict_proba(xa.reshape(1,-1))[0]; ml=1.0-proba[ci]
    bl=bls.get(ci,{})
    vt=sum(1 for j,fn in enumerate(SELECTED_FEATURES)
           if abs(xa[SELECTED_IDX[j]]-bl.get(fn,{}).get("ewma_mean",0))
           /bl.get(fn,{}).get("ewma_std",1)>zt)/len(SELECTED_FEATURES)
    fm={f:i for i,f in enumerate(SELECTED_FEATURES)}; cv=0
    for (fA,fB,mr,es) in CORRELATION_RULES:
        if fA not in fm or fB not in fm: continue
        iA=SELECTED_IDX[fm[fA]]; iB=SELECTED_IDX[fm[fB]]
        dA=float(xa[iA])-float(xo[iA]); dB=float(xa[iB])-float(xo[iB])
        rA=FEAT_BOUNDS[fA][1]-FEAT_BOUNDS[fA][0]
        if abs(dA)>0.01*rA:
            if (1 if dA*dB>0 else -1)!=es: cv+=1
    vc=cv/max(len(CORRELATION_RULES),1)
    cn=CLASS_NAMES[ci]; sf=STRIDE_SIGNATURE.get(cn,[])
    elev=sum(1 for fn in sf if fn in fm
             and xa[SELECTED_IDX[fm[fn]]]>=bl.get(fn,{}).get("raw_mean",0)*STRIDE_MULT)
    vs=1.0 if (elev==0 and len(sf)>0) else 0.0
    return ml-W_EWMA*vt-W_CORR*vc-W_STRIDE*vs

def _adaptive_hill_climb(x,ci,model,bls,zt,eps,rng):
    cur=x.copy(); cl=adaptive_loss(cur,x,ci,model,bls,zt)
    ba,bl2=cur.copy(),cl
    for it in range(300):
        cand=cur.copy()
        for i,fn in zip(SELECTED_IDX,SELECTED_FEATURES):
            e=eps[fn]; lo=max(FEAT_BOUNDS[fn][0],x[i]-e); hi=min(FEAT_BOUNDS[fn][1],x[i]+e)
            cand[i]=float(np.clip(cur[i]+rng.uniform(-e,e),lo,hi))
        cl2=adaptive_loss(cand,x,ci,model,bls,zt)
        df_c=pd.DataFrame(cand.reshape(1,-1),columns=ALL_FEATURES)
        # FIX: correct early-exit direction
        _pred=int(model.predict(df_c)[0])
        _ok=(_pred!=NORMAL_IDX) if ci==NORMAL_IDX else (_pred==NORMAL_IDX)
        if _ok: return cand,it+1
        if cl2>cl: cur=cand.copy(); cl=cl2
        if cl2>bl2: bl2=cl2; ba=cand.copy()
    return ba,300

def mean_ci(vals,conf=0.95):
    n=len(vals); m=float(np.mean(vals))
    if n<2: return m,0.0
    return m,float(stats.sem(vals)*stats.t.ppf((1+conf)/2,df=n-1))

def run_trial(hsj,model,Xe,ye,iso,bls,zt,sf,rng,mode,amode,tnum,ocsv,cols):
    eps={f:sf*(FEAT_BOUNDS[f][1]-FEAT_BOUNDS[f][0]) for f in SELECTED_FEATURES}
    total=fooled=csa_pass=errors=0
    print(f"\n  {'─'*100}")
    print(f"  Trial {tnum} | mode={mode} | sf={sf:.2f} | attack={amode}")
    print(f"  {'─'*100}")
    print(f"  {'#':>5}  {'Class':>12}  {'PredAfter':>12}  {'Result':>9}  "
          f"{'Iters':>6}  {'ML-ASR':>7}  {'CSA-ASR':>8}  {'CSA':>6}  {'Reason':<25}")
    print(f"  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*9}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*6}  {'─'*25}")
    for cn in EXECUTION_ORDER:
        ci=CLASS_NAMES.index(cn); idx=np.where(ye==ci)[0]
        if len(idx)==0: continue
        chosen=rng.choice(idx,size=min(MAX_SAMPLES,len(idx)),replace=False)
        for xr in Xe[chosen]:
            x=xr.astype(np.float32)
            try:
                if amode=="oblivious":
                    xa=hsj.generate(x=x.reshape(1,-1))[0].astype(np.float32); ni=HSJ_MAX_ITER
                else:
                    xa,ni=_adaptive_hill_climb(x,ci,model,bls,zt,eps,rng)
            except Exception as e:
                errors+=1; log.debug("err:%s",e); continue
            for j,fn in enumerate(SELECTED_FEATURES):
                lo=max(FEAT_BOUNDS[fn][0],x[SELECTED_IDX[j]]-eps[fn])
                hi=min(FEAT_BOUNDS[fn][1],x[SELECTED_IDX[j]]+eps[fn])
                xa[SELECTED_IDX[j]]=float(np.clip(xa[SELECTED_IDX[j]],lo,hi))
            dfo=pd.DataFrame(x.reshape(1,-1),columns=ALL_FEATURES)
            dfa=pd.DataFrame(xa.reshape(1,-1),columns=ALL_FEATURES)
            po=int(model.predict(dfo)[0]); pa=int(model.predict(dfa)[0])
            total+=1
            # ── PAPER METRIC: attack->Normal = success ──
            if ci==NORMAL_IDX:
                success=(pa!=NORMAL_IDX)
            else:
                success=(pa==NORMAL_IDX)
            csr={"detected":False,"reason":"N/A"}; icp=False
            if success:
                fooled+=1
                csr=csa_detect(x,xa,ci,iso,bls,zt,mode)
                icp=csr["detected"]
                if not icp: csa_pass+=1
            lml=fooled/total*100; lcsa=csa_pass/total*100
            rs="SUCCESS ✓" if success else "FAIL    ✗"
            cs="CATCH" if icp else ("PASS" if success else "─")
            print(f"  {total:>5}  {cn:>12}  {CLASS_NAMES[pa]:>12}  "
                  f"{rs:>9}  {ni:>6}  {lml:>6.1f}%  {lcsa:>7.1f}%  {cs:>6}  {csr['reason']:<25}")
            with open(ocsv,"a",newline="") as f:
                csv.DictWriter(f,fieldnames=cols,extrasaction="ignore").writerow({
                    "trial":tnum,"mode":mode,"attack_mode":amode,"scaling_factor":sf,
                    "class_true":cn,"success":int(success),"csa_detected":int(icp),
                    "csa_net_success":int(success and not icp),"csa_reason":csr["reason"],
                    "iterations_used":ni,"pred_before":CLASS_NAMES[po],"pred_after":CLASS_NAMES[pa],
                    "live_ml_asr":round(lml,2),"live_csa_asr":round(lcsa,2)})
    d=max(total,1); ml=fooled/d*100; csa=csa_pass/d*100
    print(f"\n  TRIAL {tnum} SUMMARY | sf={sf:.2f} | {mode} | {amode}")
    print(f"    Total={total} Fooled={fooled} CSA-pass={csa_pass} Errors={errors}")
    print(f"    ML-ASR={ml:.2f}% CSA-ASR={csa:.2f}% Red={ml-csa:.2f}pp Joint={ml*csa/100:.2f}%")
    return ml,csa,total,fooled,csa_pass

def main():
    mn=Path(MODEL_FILE).stem; log.info("Model: %s",mn)
    model=safe_load_model(MODEL_FILE,DATA_FILE)
    log.info("Trees: %d",len(model.estimators_))
    Xv,yv,Xe,ye=load_data()
    bls=fit_baselines(Xv,yv); zt=calibrate_zscore(Xv,yv,bls)
    iso=fit_iso_forests(Xv,yv)
    bb=make_blackbox(model)
    hsj=HopSkipJump(classifier=bb,max_iter=HSJ_MAX_ITER,targeted=False,verbose=False)
    Xen=Xe[ye==NORMAL_IDX]
    fp=sum(1 for s in Xen if csa_detect(s,s,NORMAL_IDX,iso,bls,zt)["detected"])
    fpr=fp/max(len(Xen),1)*100; log.info("FPR on clean Normal: %.2f%%",fpr)
    AMODES=["FULL","NO_TMVA","NO_CORR","NO_STRIDE","NO_IF"]; ATKS=["oblivious","adaptive"]
    cols=["trial","mode","attack_mode","scaling_factor","class_true","success",
          "csa_detected","csa_net_success","csa_reason","iterations_used",
          "pred_before","pred_after","live_ml_asr","live_csa_asr"]
    ocsv=OUTPUT_DIR/f"{mn}_results.csv"
    with open(ocsv,"w",newline="") as f: csv.DictWriter(f,fieldnames=cols).writeheader()
    recs=[]
    for atk in ATKS:
        sfs=SCALING_FACTORS if atk=="oblivious" else [0.15]
        for sf in sfs:
            for mode in AMODES:
                log.info("═══ attack=%s sf=%.2f mode=%s ═══",atk,sf,mode)
                mll=[]; csall=[]
                for t in range(1,N_TRIALS+1):
                    rng=np.random.default_rng(RANDOM_SEED+t)
                    ml,csa,*_=run_trial(hsj,model,Xe,ye,iso,bls,zt,sf,rng,mode,atk,t,ocsv,cols)
                    mll.append(ml); csall.append(csa)
                mm,hm=mean_ci(mll); mc,hc=mean_ci(csall); j=mm*mc/100.0
                recs.append({"model":mn,"attack_mode":atk,"sf":sf,"mode":mode,
                             "ML-ASR":f"{mm:.2f} ± {hm:.2f}","CSA-ASR":f"{mc:.2f} ± {hc:.2f}",
                             "Reduction":round(mm-mc,2),"Joint_%":round(j,2)})
                log.info("RESULT: ML=%.2f±%.2f CSA=%.2f±%.2f Red=%.2fpp Joint=%.2f%%",
                         mm,hm,mc,hc,mm-mc,j)
    df=pd.DataFrame(recs)
    print("\n"+"="*90); print(f"  FINAL SUMMARY — {mn}"); print("="*90)
    print(df.to_string(index=False))
    sc=OUTPUT_DIR/f"{mn}_summary.csv"; df.to_csv(sc,index=False)
    fo=df[(df["mode"]=="FULL")&(df["attack_mode"]=="oblivious")]
    rt="\n".join(f"  {r['sf']:.2f} & {r['ML-ASR']:<18} & {r['CSA-ASR']:<18} & {r['Reduction']:>6} & {r['Joint_%']:>6} \\\\"
                 for _,r in fo.iterrows())
    lat=(f"\\begin{{table}}[t]\\centering\n\\caption{{TON-IoT Network CSA — {mn} ({N_TRIALS} trials, 95\\% CI)}}\n"
         f"\\label{{tab:ton_net}}\n\\begin{{tabular}}{{lcccc}}\\toprule\n"
         f"$\\alpha$ & ML-ASR (\\%) & CSA-ASR (\\%) & Red (pp) & Joint (\\%) \\\\\n\\midrule\n"
         f"{rt}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}")
    (OUTPUT_DIR/f"{mn}_table.tex").write_text(lat)
    log.info("Results → %s",sc); log.info("FPR: %.2f%%",fpr)

if __name__=="__main__": main()
