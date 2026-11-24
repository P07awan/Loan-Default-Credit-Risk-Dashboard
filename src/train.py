"""Train, compare and evaluate default-prediction models; export everything the dashboards need."""
import json, os, sys
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from features import TARGET, add_features, load_clean, model_columns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def seg_table(df, col, bins=None, labels=None, order=None):
    s = pd.cut(df[col], bins=bins, labels=labels, right=False) if bins is not None else df[col]
    g = df.groupby(s, observed=True)[TARGET].agg(["count", "mean"]).reset_index()
    rows = [{"segment": str(r[col]), "customers": int(r["count"]), "default_rate": round(float(r["mean"]), 4)}
            for _, r in g.iterrows()]
    return sorted(rows, key=lambda r: order.index(r["segment"])) if order else rows


def main():
    raw = load_clean(os.path.join(ROOT, "data", "UCI_Credit_Card.csv"))
    df = add_features(raw)
    cols = model_columns(df)
    X, y = df[cols], df[TARGET]

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250,
                                                            l2_regularization=1.0, random_state=SEED),
    }
    val_auc, fitted = {}, {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        fitted[name] = m
        val_auc[name] = roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])
        print(f"{name:20s} val AUC = {val_auc[name]:.4f}")
    best = max(val_auc, key=val_auc.get)
    print("Selected:", best)

    # decision threshold: maximise F1 on validation set
    pv = fitted[best].predict_proba(X_va)[:, 1]
    ths = np.linspace(0.05, 0.9, 86)
    thr = float(ths[np.argmax([f1_score(y_va, pv >= t) for t in ths])])

    metrics, roc_pts = {}, {}
    for name, m in fitted.items():
        p = m.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, p)
        idx = np.linspace(0, len(fpr) - 1, 60).astype(int)
        roc_pts[name] = {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()}
        metrics[name] = {"roc_auc": round(roc_auc_score(y_te, p), 4),
                         "pr_auc": round(average_precision_score(y_te, p), 4),
                         "ks": round(ks_stat(y_te, p), 4)}

    p_te = fitted[best].predict_proba(X_te)[:, 1]
    pred = p_te >= thr
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    metrics[best].update({"threshold": round(thr, 2), "precision": round(precision_score(y_te, pred), 4),
                          "recall": round(recall_score(y_te, pred), 4), "f1": round(f1_score(y_te, pred), 4)})

    # risk bands on test set (deciles of predicted PD)
    t = pd.DataFrame({"p": p_te, "y": y_te.values})
    t["decile"] = pd.qcut(t["p"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    base = t["y"].mean()
    dec = t.groupby("decile").agg(customers=("y", "size"), avg_pd=("p", "mean"), default_rate=("y", "mean")).reset_index()
    dec["lift"] = dec["default_rate"] / base
    deciles = [{k: (round(float(v), 4) if k not in ("decile", "customers") else int(v)) for k, v in r.items()}
               for r in dec.to_dict("records")]

    # permutation importance (on a test subsample for speed)
    sub = X_te.sample(2500, random_state=SEED)
    pi = permutation_importance(fitted[best], sub, y_te.loc[sub.index], scoring="roc_auc",
                                n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = sorted(zip(cols, pi.importances_mean), key=lambda x: -x[1])[:12]
    importance = [{"feature": f, "importance": round(float(v), 4)} for f, v in imp]

    d = raw.copy()
    seg = {
        "Repayment status (last month)": seg_table(d.assign(PAY_1=d["PAY_1"].clip(-1, 4).map(
            {-1: "0 Paid duly", 0: "1 Revolving", 1: "2 1 mo late", 2: "3 2 mo late", 3: "4 3 mo late", 4: "5 4+ mo late"}
        ).str[2:]), "PAY_1",
            order=["Paid duly", "Revolving", "1 mo late", "2 mo late", "3 mo late", "4+ mo late"]),
        "Credit limit": seg_table(d, "LIMIT_BAL", [0, 50e3, 100e3, 200e3, 400e3, 2e6],
                                  ["<50K", "50–100K", "100–200K", "200–400K", "400K+"]),
        "Age band": seg_table(d, "AGE", [20, 30, 40, 50, 60, 90], ["20s", "30s", "40s", "50s", "60+"]),
        "Education": seg_table(d.assign(EDUCATION=d["EDUCATION"].map(
            {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"})), "EDUCATION",
                              order=["Graduate school", "University", "High school", "Other"]),
    }
    out = {
        "summary": {"customers": int(len(df)), "defaults": int(y.sum()), "default_rate": round(float(y.mean()), 4),
                    "avg_limit": round(float(df["LIMIT_BAL"].mean()), 0),
                    "avg_utilization": round(float(df["AVG_UTILIZATION"].mean()), 4),
                    "best_model": best, "test_size": int(len(y_te))},
        "metrics": metrics, "roc": roc_pts, "deciles": deciles, "importance": importance, "segments": seg,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    with open(os.path.join(ROOT, "reports", "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    joblib.dump({"model": fitted[best], "columns": cols, "threshold": thr, "name": best},
                os.path.join(ROOT, "models", "best_model.joblib"), compress=3)
    print(json.dumps(metrics[best], indent=1))


if __name__ == "__main__":
    main()
