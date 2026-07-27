import re, numpy as np, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

d = joblib.load("models/ablation_inputs.joblib")
Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
print("train", Xtr.shape, "test", Xte.shape)
print("train classes:", sorted(set(map(str, ytr[:5000]))))

meta = joblib.load("models/preprocessing_meta.joblib")
tags = list(meta["top_tags"])
lo   = len(meta["numeric_features"]) + meta["n_svd_components"] + meta["n_clusters"]
hi   = lo + len(tags)

norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
gn   = [norm(x) for x in meta["genre_classes"]]
isg  = np.array([any(x and (x in norm(t) or norm(t) in x) for x in gn) for t in tags])
print(f"{isg.sum()} of {len(tags)} tags name a genre class\n")

masks = {"all features (as submitted)": np.ones(hi, bool)}
m = np.ones(hi, bool); m[lo:hi] = False; masks["no tags - lyrics + audio only"] = m
m = np.ones(hi, bool); m[lo:hi] = ~isg;  masks["tags minus genre-name tags"]   = m

for label, mask in masks.items():
    try:
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1)
        clf.fit(Xtr[:, mask], ytr)
        p = clf.predict(Xte[:, mask])
        print(f"{label:34} n={mask.sum():>3} acc={accuracy_score(yte,p):.4f} "
              f"macroF1={f1_score(yte,p,average='macro'):.4f}", flush=True)
    except Exception as e:
        print(f"{label:34} FAILED: {e}", flush=True)