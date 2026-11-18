"""End-to-end pipeline: EDA -> features -> models -> evaluation -> explainability.

Run from the project root:  python src/train.py
Outputs: models/, reports/figures/, reports/metrics.json, data/scored_customers.csv
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import SCALE_COLS, TARGET, feature_names, prepare  # noqa: E402

SEED = 42
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)

INK, MUTED, GRID = "#1f2933", "#6b7280", "#e5e7eb"
C_RET, C_CH = "#3b6ea8", "#d1495b"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6,
                     "font.size": 10, "axes.titleweight": "bold"})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- data
df = pd.read_csv(ROOT / "data" / "European_Bank.csv")
quality = {"rows": len(df), "columns": df.shape[1],
           "missing_values": int(df.isna().sum().sum()),
           "duplicate_rows": int(df.duplicated().sum()),
           "duplicate_customer_ids": int(df.CustomerId.duplicated().sum()),
           "churn_rate": round(float(df[TARGET].mean()), 4)}
df = df.dropna().drop_duplicates()

# ---------------------------------------------------------------- EDA
eda = {}
def rate(col):
    g = df.groupby(col)[TARGET].agg(["mean", "count"])
    return {str(k): {"churn_rate": round(v["mean"], 4), "n": int(v["count"])}
            for k, v in g.iterrows()}

for c in ["Geography", "Gender", "NumOfProducts", "IsActiveMember", "HasCrCard", "Tenure"]:
    eda[c] = rate(c)
df["AgeBand"] = pd.cut(df.Age, [17, 29, 39, 49, 59, 100],
                       labels=["18-29", "30-39", "40-49", "50-59", "60+"])
df["BalanceBand"] = pd.cut(df.Balance, [-1, 0, 50000, 100000, 150000, 1e9],
                           labels=["0", "0-50k", "50-100k", "100-150k", "150k+"])
df["CreditBand"] = pd.cut(df.CreditScore, [0, 579, 669, 739, 799, 900],
                          labels=["<580", "580-669", "670-739", "740-799", "800+"])
for c in ["AgeBand", "BalanceBand", "CreditBand"]:
    eda[c] = rate(c)
eda["active_x_products"] = {
    f"active={a}|products={p}": {"churn_rate": round(g[TARGET].mean(), 4), "n": len(g)}
    for (a, p), g in df.groupby(["IsActiveMember", "NumOfProducts"])}
eda["geo_x_active"] = {
    f"{geo}|active={a}": round(g[TARGET].mean(), 4)
    for (geo, a), g in df.groupby(["Geography", "IsActiveMember"])}
eda["means_by_class"] = df.groupby(TARGET)[["Age", "Balance", "CreditScore", "Tenure",
                                            "EstimatedSalary", "NumOfProducts"]].mean().round(2).to_dict()

# EDA figures
fig, ax = plt.subplots(figsize=(4.2, 3.2))
vc = df[TARGET].value_counts().sort_index()
ax.bar(["Retained", "Churned"], vc.values, color=[C_RET, C_CH], width=0.55)
for i, v in enumerate(vc.values):
    ax.text(i, v + 80, f"{v:,}\n({v/len(df):.1%})", ha="center", color=INK, fontsize=9)
ax.set_ylim(0, vc.max() * 1.18); ax.set_title("Class balance"); ax.grid(axis="x")
save(fig, "01_class_balance.png")

fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
for ax, c, title in zip(axes.flat,
                        ["Geography", "Gender", "NumOfProducts", "IsActiveMember", "AgeBand", "BalanceBand"],
                        ["Geography", "Gender", "Number of products", "Active member", "Age band", "Balance band"]):
    r = df.groupby(c, observed=True)[TARGET].mean()
    ax.bar(r.index.astype(str), r.values, color=C_CH, width=0.6)
    ax.axhline(df[TARGET].mean(), color=MUTED, ls="--", lw=1)
    for i, v in enumerate(r.values):
        ax.text(i, v + 0.015, f"{v:.0%}", ha="center", fontsize=8, color=INK)
    ax.set_title(f"Churn rate by {title.lower()}"); ax.set_ylim(0, min(1.08, r.max() + 0.15))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.grid(axis="x")
fig.suptitle("Churn rate by segment (dashed line = portfolio average 20.4%)", color=INK)
save(fig, "02_churn_by_segment.png")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, c in zip(axes, ["Age", "Balance", "CreditScore"]):
    for lab, col, name in [(0, C_RET, "Retained"), (1, C_CH, "Churned")]:
        ax.hist(df.loc[df[TARGET] == lab, c], bins=35, alpha=0.55, color=col, density=True, label=name)
    ax.set_title(f"{c} distribution"); ax.set_yticks([])
axes[0].legend(frameon=False)
save(fig, "03_numeric_distributions.png")

piv = df.pivot_table(index="IsActiveMember", columns="NumOfProducts", values=TARGET, aggfunc="mean")
fig, ax = plt.subplots(figsize=(5.5, 2.8))
im = ax.imshow(piv.values, cmap="Reds", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(piv.shape[1]), piv.columns); ax.set_yticks([0, 1], ["Inactive", "Active"])
ax.set_xlabel("Number of products"); ax.grid(False)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", color="white" if v > 0.55 else INK)
ax.set_title("Churn rate: engagement x product holding")
save(fig, "04_engagement_product_heatmap.png")

num = df[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
          "IsActiveMember", "EstimatedSalary", TARGET]].corr()
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.imshow(num.values, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
ax.set_xticks(range(len(num)), num.columns, rotation=60, ha="right"); ax.set_yticks(range(len(num)), num.columns)
for i in range(len(num)):
    for j in range(len(num)):
        ax.text(j, i, f"{num.values[i,j]:.2f}", ha="center", va="center", fontsize=7)
ax.set_title("Correlation matrix")
save(fig, "05_correlation.png")

# ---------------------------------------------------------------- split
X = prepare(df)
y = df[TARGET].values
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X, y, df.index, test_size=0.2, stratify=y, random_state=SEED)
split = {"train_rows": len(X_tr), "test_rows": len(X_te),
         "train_churn_rate": round(float(y_tr.mean()), 4), "test_churn_rate": round(float(y_te.mean()), 4)}

passthrough = [c for c in feature_names() if c not in SCALE_COLS]
scaler = ColumnTransformer([("num", StandardScaler(), SCALE_COLS),
                            ("pass", "passthrough", passthrough)])
spw = (y_tr == 0).sum() / (y_tr == 1).sum()

models = {
    "Logistic Regression": Pipeline([("prep", scaler),
                                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=30,
                                            class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                                    subsample=0.8, random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
                             scale_pos_weight=spw, eval_metric="auc", random_state=SEED, n_jobs=-1),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results, probas = {}, {}
for name, m in models.items():
    cv_auc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    yhat = (p >= 0.5).astype(int)
    probas[name] = p
    results[name] = {"cv_auc_mean": round(float(cv_auc.mean()), 4), "cv_auc_std": round(float(cv_auc.std()), 4),
                     "accuracy": round(accuracy_score(y_te, yhat), 4),
                     "precision": round(precision_score(y_te, yhat), 4),
                     "recall": round(recall_score(y_te, yhat), 4),
                     "f1": round(f1_score(y_te, yhat), 4),
                     "roc_auc": round(roc_auc_score(y_te, p), 4)}
    print(f"{name:20s} {results[name]}")

best_name = max(results, key=lambda k: results[k]["roc_auc"])
best = models[best_name]
p_best = probas[best_name]

# ---------------------------------------------------------------- threshold tuning
# Choose the threshold on out-of-fold TRAIN predictions (not the test set) to
# maximise F1, which trades false alarms against missed churners.
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
oof = cross_val_predict(clone(best), X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
pr, rc, th = precision_recall_curve(y_tr, oof)
f1s = 2 * pr * rc / (pr + rc + 1e-9)
thr_f1 = float(th[np.argmax(f1s[:-1])])
# Precision-first threshold: highest recall subject to precision >= 0.70
ok = np.where(pr[:-1] >= 0.70)[0]
thr_prec = float(th[ok[np.argmax(rc[ok])]]) if len(ok) else thr_f1

def at(thr):
    yhat = (p_best >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, yhat).ravel()
    return {"threshold": round(thr, 3), "accuracy": round(accuracy_score(y_te, yhat), 4),
            "precision": round(precision_score(y_te, yhat), 4), "recall": round(recall_score(y_te, yhat), 4),
            "f1": round(f1_score(y_te, yhat), 4), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "flagged": int(tp + fp)}

thresholds = {"default_0.50": at(0.5), "f1_optimal": at(thr_f1), "precision_70": at(thr_prec)}
print(best_name, thresholds)

# Lift / capture by decile on test set
order = np.argsort(-p_best)
dec = np.array_split(order, 10)
lift = [{"decile": i + 1, "min_prob": round(float(p_best[d].min()), 3),
         "churn_rate": round(float(y_te[d].mean()), 4),
         "cum_capture": round(float(y_te[np.concatenate(dec[:i + 1])].sum() / y_te.sum()), 4)}
        for i, d in enumerate(dec)]

# ---------------------------------------------------------------- evaluation figures
pal = {"Logistic Regression": "#9aa5b1", "Decision Tree": "#e0a458",
       "Random Forest": "#5b8e7d", "Gradient Boosting": "#3b6ea8", "XGBoost": "#d1495b"}
fig, ax = plt.subplots(figsize=(5.2, 4.4))
for n, p in probas.items():
    fpr, tpr, _ = roc_curve(y_te, p)
    ax.plot(fpr, tpr, color=pal[n], lw=2 if n == best_name else 1.3,
            label=f"{n} (AUC {results[n]['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], color=GRID, ls="--")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves (hold-out test set)"); ax.legend(frameon=False, fontsize=8, loc="lower right")
save(fig, "06_roc_curves.png")

fig, ax = plt.subplots(figsize=(5.2, 4.4))
for n, p in probas.items():
    pr_, rc_, _ = precision_recall_curve(y_te, p)
    ax.plot(rc_, pr_, color=pal[n], lw=2 if n == best_name else 1.3, label=n)
ax.axhline(y_te.mean(), color=GRID, ls="--")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall curves")
ax.legend(frameon=False, fontsize=8)
save(fig, "07_pr_curves.png")

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
for ax, (k, lab) in zip(axes, [("default_0.50", "Threshold 0.50"), ("f1_optimal", f"F1-optimal ({thr_f1:.2f})")]):
    t = thresholds[k]; cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
    ax.imshow(cm, cmap="Blues"); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.5 else INK, fontsize=12)
    ax.set_xticks([0, 1], ["Pred: stay", "Pred: churn"]); ax.set_yticks([0, 1], ["Actual: stay", "Actual: churn"])
    ax.set_title(f"{best_name} — {lab}", fontsize=9)
save(fig, "08_confusion_matrices.png")

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.bar([d["decile"] for d in lift], [d["churn_rate"] for d in lift], color=C_CH, width=0.65)
ax.axhline(y_te.mean(), color=MUTED, ls="--", lw=1)
ax.set_xticks(range(1, 11)); ax.set_xlabel("Risk decile (1 = highest predicted risk)")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_title("Actual churn rate by predicted-risk decile"); ax.grid(axis="x")
save(fig, "09_lift_by_decile.png")

# ---------------------------------------------------------------- explainability
fn = feature_names()
if hasattr(best, "feature_importances_"):
    imp = pd.Series(best.feature_importances_, index=fn).sort_values(ascending=False)
else:
    imp = pd.Series(np.abs(best.named_steps["clf"].coef_[0]), index=fn).sort_values(ascending=False)

lr = models["Logistic Regression"].named_steps["clf"]
lr_cols = SCALE_COLS + passthrough
lr_coef = pd.Series(lr.coef_[0], index=lr_cols).sort_values()

explainer = shap.TreeExplainer(best)
X_shap = X_te.sample(1000, random_state=SEED)
sv = explainer.shap_values(X_shap)
if isinstance(sv, list):
    sv = sv[1]
shap_imp = pd.Series(np.abs(sv).mean(0), index=fn).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(6, 5))
s = shap_imp.head(15)[::-1]
ax.barh(s.index, s.values, color=C_RET); ax.grid(axis="y")
ax.set_xlabel("Mean |SHAP value| (log-odds impact)"); ax.set_title(f"Global drivers — {best_name} (SHAP)")
save(fig, "10_shap_importance.png")

plt.figure()
shap.summary_plot(sv, X_shap, max_display=15, show=False, plot_size=(7.5, 5.5))
plt.title("SHAP beeswarm: direction and spread of each driver")
plt.savefig(FIG / "11_shap_beeswarm.png", bbox_inches="tight", dpi=130); plt.close("all")

fig, ax = plt.subplots(figsize=(6, 5))
c = lr_coef[lr_coef.abs().sort_values().index][-14:]
ax.barh(c.index, c.values, color=[C_CH if v > 0 else C_RET for v in c.values]); ax.grid(axis="y")
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_xlabel("Standardised coefficient (+ raises churn odds)"); ax.set_title("Logistic regression coefficients")
save(fig, "12_logreg_coefficients.png")

fig, ax = plt.subplots(2, 3, figsize=(11, 6))
PartialDependenceDisplay.from_estimator(best, X_te.astype(float), ["Age", "NumOfProducts", "IsActiveMember",
                                                     "Balance", "CreditScore", "Geography_Germany"],
                                        ax=ax.flat, line_kw={"color": C_CH, "lw": 2})
fig.suptitle(f"Partial dependence — {best_name}", color=INK)
save(fig, "13_partial_dependence.png")

# ---------------------------------------------------------------- score all customers
p_all = best.predict_proba(X)[:, 1]
scored = df[["CustomerId", "Surname", "Geography", "Gender", "Age", "Tenure", "Balance",
             "NumOfProducts", "IsActiveMember", "CreditScore", "EstimatedSalary", "HasCrCard", TARGET]].copy()
scored["ChurnProbability"] = p_all.round(4)
scored["RiskBand"] = pd.cut(p_all, [-0.01, 0.3, 0.6, 1.01], labels=["Low", "Medium", "High"])
scored["ChurnFlag"] = (p_all >= thr_f1).astype(int)
scored["InTestSet"] = scored.index.isin(idx_te).astype(int)
scored.sort_values("ChurnProbability", ascending=False).to_csv(ROOT / "data" / "scored_customers.csv", index=False)
bands = scored.groupby("RiskBand", observed=True).agg(customers=("CustomerId", "size"),
                                                     actual_churn=(TARGET, "mean"),
                                                     balance=("Balance", "sum")).round(4)

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.hist(p_best[y_te == 0], bins=40, alpha=0.6, color=C_RET, label="Retained", density=True)
ax.hist(p_best[y_te == 1], bins=40, alpha=0.6, color=C_CH, label="Churned", density=True)
ax.axvline(thr_f1, color=INK, ls="--", lw=1); ax.text(thr_f1 + 0.01, ax.get_ylim()[1] * 0.9, f"threshold {thr_f1:.2f}", fontsize=8)
ax.set_xlabel("Predicted churn probability"); ax.set_yticks([]); ax.legend(frameon=False)
ax.set_title("Score separation on test set")
save(fig, "14_probability_distribution.png")

# ---------------------------------------------------------------- persist
joblib.dump({"model": best, "name": best_name, "features": fn, "threshold": thr_f1,
             "threshold_precision70": thr_prec}, ROOT / "models" / "best_model.joblib")
joblib.dump(models["Logistic Regression"], ROOT / "models" / "logistic_regression.joblib")

metrics = {"data_quality": quality, "split": split, "models": results, "best_model": best_name,
           "thresholds": thresholds, "lift": lift,
           "feature_importance": imp.round(4).to_dict(),
           "shap_importance": shap_imp.round(4).to_dict(),
           "logreg_coefficients": lr_coef.round(4).to_dict(),
           "risk_bands": {str(k): v for k, v in bands.to_dict("index").items()},
           "eda": eda}
(ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
print("best:", best_name, "thr_f1", thr_f1, "thr_p70", thr_prec)
print(shap_imp.head(10)); print(bands)
