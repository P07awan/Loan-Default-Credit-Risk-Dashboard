"""Streamlit dashboard:  streamlit run app.py"""
import json, os, sys
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from features import load_clean
from predict import load_model, score

st.set_page_config(page_title="Loan Default & Credit Risk Dashboard", layout="wide")
R = json.load(open(os.path.join(ROOT, "reports", "results.json")))
S, M = R["summary"], R["metrics"][R["summary"]["best_model"]]


@st.cache_data
def data():
    return load_clean(os.path.join(ROOT, "data", "UCI_Credit_Card.csv"))


@st.cache_resource
def model():
    return load_model()


st.title("Loan Default & Credit Risk Dashboard")
st.caption("UCI Credit Card Default dataset · 30,000 cardholders · predicting next-month default")
t1, t2, t3 = st.tabs(["Portfolio overview", "Model performance", "Score a customer"])

with t1:
    c = st.columns(4)
    c[0].metric("Customers", f'{S["customers"]:,}')
    c[1].metric("Default rate", f'{S["default_rate"]:.1%}')
    c[2].metric("Avg credit limit (NT$)", f'{S["avg_limit"]:,.0f}')
    c[3].metric("Avg utilisation", f'{S["avg_utilization"]:.0%}')
    seg = st.selectbox("Segment by", list(R["segments"]))
    sdf = pd.DataFrame(R["segments"][seg]).set_index("segment")
    a, b = st.columns([2, 1])
    a.bar_chart(sdf["default_rate"], y_label="Default rate")
    b.dataframe(sdf.style.format({"default_rate": "{:.1%}", "customers": "{:,}"}), use_container_width=True)

with t2:
    c = st.columns(5)
    c[0].metric("Selected model", S["best_model"])
    c[1].metric("ROC-AUC", f'{M["roc_auc"]:.3f}')
    c[2].metric("KS", f'{M["ks"]:.3f}')
    c[3].metric("Recall", f'{M["recall"]:.0%}')
    c[4].metric("Precision", f'{M["precision"]:.0%}')
    a, b = st.columns(2)
    dec = pd.DataFrame(R["deciles"]).set_index("decile")
    a.subheader("Default rate by risk decile"); a.bar_chart(dec["default_rate"])
    imp = pd.DataFrame(R["importance"]).set_index("feature")
    b.subheader("Top drivers (permutation importance)"); b.bar_chart(imp["importance"], horizontal=True)
    st.subheader("Model comparison")
    st.dataframe(pd.DataFrame(R["metrics"]).T[["roc_auc", "pr_auc", "ks"]].round(3), use_container_width=True)
    st.subheader("Decile table")
    st.dataframe(dec.style.format({"avg_pd": "{:.1%}", "default_rate": "{:.1%}", "lift": "{:.2f}x"}), use_container_width=True)

with t3:
    df = data()
    st.write("Pick a customer from the dataset, optionally edit the inputs, and see the model's estimated probability of default.")
    cid = st.number_input("Customer ID", 1, int(df["ID"].max()), 1)
    row = df[df["ID"] == cid].iloc[0].copy()
    with st.expander("Edit inputs", expanded=True):
        c = st.columns(3)
        row["LIMIT_BAL"] = c[0].number_input("Credit limit", 0, 2_000_000, int(row["LIMIT_BAL"]), 10_000)
        row["AGE"] = c[1].number_input("Age", 18, 100, int(row["AGE"]))
        row["EDUCATION"] = c[2].selectbox("Education", [1, 2, 3, 4], int(row["EDUCATION"]) - 1,
                                          format_func={1: "Graduate", 2: "University", 3: "High school", 4: "Other"}.get)
        st.caption("Repayment status per month (−2 no use, −1 paid duly, 0 revolving, 1–8 months late). Most recent first.")
        c = st.columns(6)
        for i, col in enumerate(c):
            k = f"PAY_{i + 1}"; row[k] = col.number_input(k, -2, 8, int(row[k]))
    res = score(pd.DataFrame([row]), model()).iloc[0]
    c = st.columns(3)
    c[0].metric("Probability of default", f'{res["pd"]:.1%}')
    c[1].metric("Risk band", res["risk_band"])
    c[2].metric("Flagged high-risk", "Yes" if res["flag_high_risk"] else "No")
    st.caption(f'Actual outcome in dataset for original record: {"default" if df[df["ID"] == cid].iloc[0]["DEFAULT"] else "no default"}')
