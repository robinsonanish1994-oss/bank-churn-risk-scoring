"""Streamlit app: Bank Customer Churn — Predictive Risk Scoring.

Run:  streamlit run app.py
Requires models/best_model.joblib, reports/metrics.json and
data/scored_customers.csv (produced by `python src/train.py`).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from features import prepare  # noqa: E402

st.set_page_config(page_title="Churn Risk Intelligence", page_icon="🏦", layout="wide")

RED, BLUE, GREY, AMBER = "#d1495b", "#3b6ea8", "#9aa5b1", "#e0a458"


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "best_model.joblib")


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data
def load_data():
    scored = pd.read_csv(ROOT / "data" / "scored_customers.csv")
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    return scored, metrics


bundle = load_model()
model, THR = bundle["model"], bundle["threshold"]
explainer = load_explainer(model)
scored, metrics = load_data()


def predict(row: dict) -> float:
    return float(model.predict_proba(prepare(pd.DataFrame([row])))[:, 1][0])


def band(p):
    if p >= 0.6:
        return "High", RED
    if p >= 0.3:
        return "Medium", AMBER
    return "Low", BLUE


def gauge(p, title="Churn probability"):
    lab, col = band(p)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "valueformat": ".1f"},
        title={"text": f"{title}<br><span style='font-size:0.8em;color:{col}'>{lab} risk</span>"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": col},
               "steps": [{"range": [0, 30], "color": "#eef3f9"},
                         {"range": [30, 60], "color": "#fbf1e3"},
                         {"range": [60, 100], "color": "#f9e3e6"}],
               "threshold": {"line": {"color": "#1f2933", "width": 3}, "value": THR * 100}}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def customer_form(prefix: str, defaults: dict):
    c1, c2, c3 = st.columns(3)
    with c1:
        geo = st.selectbox("Geography", ["France", "Germany", "Spain"],
                           index=["France", "Germany", "Spain"].index(defaults["Geography"]), key=prefix + "geo")
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True,
                          index=["Female", "Male"].index(defaults["Gender"]), key=prefix + "gen")
        age = st.slider("Age", 18, 92, defaults["Age"], key=prefix + "age")
        tenure = st.slider("Tenure (years)", 0, 10, defaults["Tenure"], key=prefix + "ten")
    with c2:
        credit = st.slider("Credit score", 350, 850, defaults["CreditScore"], key=prefix + "cs")
        balance = st.number_input("Balance (€)", 0.0, 260000.0, float(defaults["Balance"]), 1000.0, key=prefix + "bal")
        salary = st.number_input("Estimated salary (€)", 10.0, 200000.0, float(defaults["EstimatedSalary"]),
                                 1000.0, key=prefix + "sal")
    with c3:
        products = st.select_slider("Number of products", [1, 2, 3, 4], defaults["NumOfProducts"], key=prefix + "np")
        active = st.toggle("Active member", bool(defaults["IsActiveMember"]), key=prefix + "act")
        card = st.toggle("Has credit card", bool(defaults["HasCrCard"]), key=prefix + "cc")
    return {"CreditScore": credit, "Geography": geo, "Gender": gender, "Age": age, "Tenure": tenure,
            "Balance": balance, "NumOfProducts": products, "HasCrCard": int(card),
            "IsActiveMember": int(active), "EstimatedSalary": salary}


DEFAULT = {"CreditScore": 650, "Geography": "Germany", "Gender": "Female", "Age": 48, "Tenure": 4,
           "Balance": 110000.0, "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 0,
           "EstimatedSalary": 95000.0}

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🏦 Churn Risk Intelligence")
page = st.sidebar.radio("Module", ["Portfolio overview", "Risk calculator", "What-if simulator",
                                   "Probability distribution", "Feature importance", "Model performance",
                                   "Customer lookup"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Model: **{bundle['name']}**  \nTest ROC-AUC: **{metrics['models'][bundle['name']]['roc_auc']:.3f}**"
                   f"  \nDecision threshold: **{THR:.2f}** (F1-optimal)")

# ------------------------------------------------------------------ pages
if page == "Portfolio overview":
    st.title("Portfolio churn risk overview")
    k = st.columns(5)
    k[0].metric("Customers", f"{len(scored):,}")
    k[1].metric("Historical churn rate", f"{scored.Exited.mean():.1%}")
    hi = scored[scored.RiskBand == "High"]
    k[2].metric("High-risk customers", f"{len(hi):,}")
    k[3].metric("Balance at high risk", f"€{hi.Balance.sum()/1e6:,.1f}M")
    k[4].metric("Flagged for retention", f"{scored.ChurnFlag.sum():,}")

    c1, c2 = st.columns(2)
    with c1:
        b = scored.groupby("RiskBand").agg(customers=("CustomerId", "size"), churn=("Exited", "mean")).reindex(
            ["Low", "Medium", "High"]).reset_index()
        fig = px.bar(b, x="RiskBand", y="customers", color="RiskBand", text="customers",
                     color_discrete_map={"Low": BLUE, "Medium": AMBER, "High": RED},
                     title="Customers by risk band")
        fig.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig, width="stretch")
        st.caption("Actual historical churn — Low: {:.0%} · Medium: {:.0%} · High: {:.0%}".format(*b.churn))
    with c2:
        g = scored.groupby(["Geography", "IsActiveMember"]).ChurnProbability.mean().reset_index()
        g["Engagement"] = g.IsActiveMember.map({0: "Inactive", 1: "Active"})
        fig = px.bar(g, x="Geography", y="ChurnProbability", color="Engagement", barmode="group",
                     color_discrete_map={"Inactive": RED, "Active": BLUE}, title="Mean predicted risk by market & engagement")
        fig.update_layout(yaxis_tickformat=".0%", height=360)
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        p = scored.pivot_table(index="IsActiveMember", columns="NumOfProducts", values="Exited", aggfunc="mean")
        fig = px.imshow(p.values, x=[str(c) for c in p.columns], y=["Inactive", "Active"], text_auto=".0%",
                        color_continuous_scale="Reds", zmin=0, zmax=1,
                        labels=dict(x="Number of products", color="Churn"), title="Churn: engagement × products")
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")
    with c4:
        a = scored.assign(AgeBand=pd.cut(scored.Age, [17, 29, 39, 49, 59, 100],
                                         labels=["18-29", "30-39", "40-49", "50-59", "60+"]))
        a = a.groupby("AgeBand", observed=True).Exited.mean().reset_index()
        fig = px.bar(a, x="AgeBand", y="Exited", title="Churn by age band", color_discrete_sequence=[RED])
        fig.update_layout(yaxis_tickformat=".0%", height=320)
        st.plotly_chart(fig, width="stretch")

elif page == "Risk calculator":
    st.title("Customer churn risk calculator")
    st.write("Enter a customer's profile to get a churn probability, risk band and the factors driving it.")
    row = customer_form("calc_", DEFAULT)
    p = predict(row)
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.plotly_chart(gauge(p), width="stretch")
        flag = p >= THR
        (st.error if flag else st.success)(
            f"**Churn flag: {'YES — prioritise for retention' if flag else 'No'}** "
            f"(threshold {THR:.2f})")
        pct = (scored.ChurnProbability < p).mean()
        st.caption(f"Riskier than {pct:.0%} of the current customer base.")
    with c2:
        X = prepare(pd.DataFrame([row]))
        sv = explainer.shap_values(X)[0]
        contrib = pd.Series(sv, index=X.columns)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)[::-1]
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                               marker_color=[RED if v > 0 else BLUE for v in top.values]))
        fig.update_layout(title="Why this score? (SHAP, log-odds)", height=340,
                          xaxis_title="← lowers risk · raises risk →", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    st.subheader("Suggested retention actions")
    tips = []
    if not row["IsActiveMember"]:
        tips.append("Re-engagement journey: digital onboarding nudge, relationship-manager call, usage incentives.")
    if row["NumOfProducts"] == 1:
        tips.append("Cross-sell a second, *relevant* product (single-product customers churn ~3.7× more than two-product holders).")
    if row["NumOfProducts"] >= 3:
        tips.append("Product-fit review: 3–4 product holders churn at 83–100% — check for mis-selling, fees or bundle friction.")
    if 40 <= row["Age"] <= 65:
        tips.append("Mid-life segment: offer advisory/wealth or retirement-planning conversations.")
    if row["Geography"] == "Germany":
        tips.append("German market: benchmark pricing and service against local competitors.")
    if row["Balance"] > 100000:
        tips.append("High-value balance at stake: assign priority service / loyalty rate.")
    for t in tips or ["No major risk factor — maintain standard engagement."]:
        st.markdown(f"- {t}")

elif page == "What-if simulator":
    st.title("What-if scenario simulator")
    st.write("Start from a baseline customer, then change engagement and product levers to see how risk moves.")
    with st.expander("Baseline customer", expanded=True):
        base = customer_form("base_", DEFAULT)
    p0 = predict(base)
    st.subheader("Levers")
    l1, l2, l3, l4 = st.columns(4)
    new_active = l1.toggle("Make active member", value=bool(base["IsActiveMember"]))
    new_prod = l2.select_slider("Products held", [1, 2, 3, 4], value=base["NumOfProducts"])
    new_card = l3.toggle("Has credit card ", value=bool(base["HasCrCard"]))
    new_bal = l4.slider("Balance (€)", 0, 250000, int(base["Balance"]), 5000)
    scen = {**base, "IsActiveMember": int(new_active), "NumOfProducts": new_prod,
            "HasCrCard": int(new_card), "Balance": float(new_bal)}
    p1 = predict(scen)
    c1, c2, c3 = st.columns([1, 1, 1.2])
    c1.plotly_chart(gauge(p0, "Baseline"), width="stretch")
    c2.plotly_chart(gauge(p1, "Scenario"), width="stretch")
    with c3:
        st.metric("Change in churn probability", f"{p1:.1%}", f"{(p1 - p0) * 100:+.1f} pts", delta_color="inverse")
        rows = []
        for a in [0, 1]:
            for n in [1, 2, 3, 4]:
                rows.append({"Active": "Active" if a else "Inactive", "Products": n,
                             "p": predict({**base, "IsActiveMember": a, "NumOfProducts": n})})
        grid = pd.DataFrame(rows)
        fig = px.line(grid, x="Products", y="p", color="Active", markers=True,
                      color_discrete_map={"Active": BLUE, "Inactive": RED},
                      title="Risk across engagement × products for this customer")
        fig.update_layout(yaxis_tickformat=".0%", height=300, xaxis_dtick=1)
        st.plotly_chart(fig, width="stretch")
    st.subheader("Sensitivity to age and balance")
    s1, s2 = st.columns(2)
    ages = np.arange(18, 86)
    s1.plotly_chart(px.line(x=ages, y=[predict({**scen, "Age": int(a)}) for a in ages],
                            labels={"x": "Age", "y": "Churn probability"}, title="Age sweep (scenario)")
                    .update_layout(yaxis_tickformat=".0%", height=300).update_traces(line_color=RED),
                    width="stretch")
    bals = np.linspace(0, 250000, 51)
    s2.plotly_chart(px.line(x=bals, y=[predict({**scen, "Balance": float(b)}) for b in bals],
                            labels={"x": "Balance (€)", "y": "Churn probability"}, title="Balance sweep (scenario)")
                    .update_layout(yaxis_tickformat=".0%", height=300).update_traces(line_color=BLUE),
                    width="stretch")

elif page == "Probability distribution":
    st.title("Churn probability distribution")
    seg = st.multiselect("Geography", ["France", "Germany", "Spain"], ["France", "Germany", "Spain"])
    d = scored[scored.Geography.isin(seg)].copy()
    d["Outcome"] = d.Exited.map({0: "Retained", 1: "Churned"})
    fig = px.histogram(d, x="ChurnProbability", color="Outcome", nbins=50, barmode="overlay", opacity=0.65,
                       color_discrete_map={"Retained": BLUE, "Churned": RED},
                       title="Predicted probability by actual outcome")
    fig.add_vline(x=THR, line_dash="dash", annotation_text=f"threshold {THR:.2f}")
    st.plotly_chart(fig, width="stretch")
    thr = st.slider("Explore a different decision threshold", 0.05, 0.95, float(round(THR, 2)), 0.01)
    t = d[d.InTestSet == 1]
    yhat = (t.ChurnProbability >= thr).astype(int)
    tp = int(((yhat == 1) & (t.Exited == 1)).sum()); fp = int(((yhat == 1) & (t.Exited == 0)).sum())
    fn = int(((yhat == 0) & (t.Exited == 1)).sum())
    m = st.columns(4)
    m[0].metric("Flagged (test set)", f"{tp + fp:,}")
    m[1].metric("Precision", f"{tp / max(tp + fp, 1):.1%}")
    m[2].metric("Recall", f"{tp / max(tp + fn, 1):.1%}")
    m[3].metric("False alarms", f"{fp:,}")
    st.caption("Metrics are computed on the held-out test customers only, so they are honest out-of-sample estimates.")
    lift = pd.DataFrame(metrics["lift"])
    fig = px.bar(lift, x="decile", y="churn_rate", title="Actual churn rate by predicted-risk decile (test set)",
                 color_discrete_sequence=[RED])
    fig.update_layout(yaxis_tickformat=".0%", xaxis_dtick=1)
    st.plotly_chart(fig, width="stretch")
    st.info(f"The top 20% riskiest customers contain **{lift.cum_capture[1]:.0%}** of all churners; "
            f"the top 30% contain **{lift.cum_capture[2]:.0%}**.")

elif page == "Feature importance":
    st.title("Feature importance dashboard")
    c1, c2 = st.columns(2)
    s = pd.Series(metrics["shap_importance"]).sort_values().tail(15)
    c1.plotly_chart(px.bar(x=s.values, y=s.index, orientation="h", title="Global SHAP importance (mean |SHAP|)",
                           color_discrete_sequence=[BLUE], labels={"x": "mean |SHAP|", "y": ""}),
                    width="stretch")
    lr = pd.Series(metrics["logreg_coefficients"])
    lr = lr.reindex(lr.abs().sort_values().index).tail(15)
    c2.plotly_chart(px.bar(x=lr.values, y=lr.index, orientation="h", title="Logistic regression coefficients",
                           color=np.where(lr.values > 0, "raises risk", "lowers risk"),
                           color_discrete_map={"raises risk": RED, "lowers risk": BLUE},
                           labels={"x": "standardised coefficient", "y": "", "color": ""}),
                    width="stretch")
    st.subheader("SHAP beeswarm & partial dependence")
    st.image(str(ROOT / "reports" / "figures" / "11_shap_beeswarm.png"), width="stretch")
    st.image(str(ROOT / "reports" / "figures" / "13_partial_dependence.png"), width="stretch")

elif page == "Model performance":
    st.title("Model comparison")
    r = pd.DataFrame(metrics["models"]).T
    r.columns = ["CV AUC (mean)", "CV AUC (std)", "Accuracy", "Precision", "Recall", "F1", "Test ROC-AUC"]
    st.dataframe(r.style.format("{:.3f}").highlight_max(axis=0, color="#dbe7f3"), width="stretch")
    st.caption("Precision / recall / F1 at the default 0.50 threshold. CV = 5-fold stratified on training data.")
    st.subheader(f"Threshold strategy — {bundle['name']}")
    t = pd.DataFrame(metrics["thresholds"]).T
    st.dataframe(t, width="stretch")
    c1, c2 = st.columns(2)
    c1.image(str(ROOT / "reports" / "figures" / "06_roc_curves.png"))
    c2.image(str(ROOT / "reports" / "figures" / "08_confusion_matrices.png"))

elif page == "Customer lookup":
    st.title("Customer risk lookup")
    c1, c2, c3 = st.columns(3)
    bands = c1.multiselect("Risk band", ["High", "Medium", "Low"], ["High"])
    geos = c2.multiselect("Geography", ["France", "Germany", "Spain"], ["France", "Germany", "Spain"])
    only_active = c3.selectbox("Engagement", ["All", "Inactive only", "Active only"])
    d = scored[scored.RiskBand.isin(bands) & scored.Geography.isin(geos)]
    if only_active != "All":
        d = d[d.IsActiveMember == (1 if only_active == "Active only" else 0)]
    st.write(f"**{len(d):,}** customers · balance €{d.Balance.sum()/1e6:,.1f}M")
    st.dataframe(d.drop(columns=["InTestSet"]).head(500), width="stretch", hide_index=True)
    st.download_button("Download list (CSV)", d.to_csv(index=False), "churn_target_list.csv", "text/csv")
