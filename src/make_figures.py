"""Render README figures from reports/results.json."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "reports", "results.json")))
OUT = os.path.join(ROOT, "reports", "figures")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C = {"Logistic Regression": "#9aa5b1", "Random Forest": "#4f8cc9", "Gradient Boosting": "#d9480f"}

fig, ax = plt.subplots(figsize=(5, 4.2))
for n, r in R["roc"].items():
    ax.plot(r["fpr"], r["tpr"], color=C[n], lw=2, label=f'{n} (AUC {R["metrics"][n]["roc_auc"]:.3f})')
ax.plot([0, 1], [0, 1], "--", color="#ccc"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves (hold-out test set)"); ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/roc.png", dpi=150); plt.close(fig)

d = R["deciles"]; base = R["summary"]["default_rate"] * 100
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.bar([x["decile"] for x in d], [x["default_rate"] * 100 for x in d], color="#d9480f")
ax.axhline(base, ls="--", color="#555", lw=1); ax.text(1, base + 1.5, f"portfolio avg {base:.1f}%", fontsize=8)
ax.set_xticks(range(1, 11)); ax.set_xlabel("Risk decile (1 = lowest predicted PD)"); ax.set_ylabel("Actual default rate (%)")
ax.set_title("Default rate by model risk decile"); fig.tight_layout(); fig.savefig(f"{OUT}/deciles.png", dpi=150); plt.close(fig)

imp = R["importance"][::-1]
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.barh([i["feature"] for i in imp], [i["importance"] for i in imp], color="#4f8cc9")
ax.set_xlabel("Drop in AUC when feature is shuffled"); ax.set_title("Permutation importance (top 12)")
fig.tight_layout(); fig.savefig(f"{OUT}/importance.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, k in zip(axs, ["Repayment status (last month)", "Credit limit"]):
    s = R["segments"][k]
    ax.bar([x["segment"] for x in s], [x["default_rate"] * 100 for x in s], color="#d9480f")
    ax.axhline(base, ls="--", color="#555", lw=1); ax.set_title(k); ax.set_ylabel("Default rate (%)")
fig.tight_layout(); fig.savefig(f"{OUT}/segments.png", dpi=150); plt.close(fig)
print("figures ok")
