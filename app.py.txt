import streamlit as st
import pandas as pd
import numpy as np
import json, joblib

st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("best_attrition_model.joblib")  # trained pipeline (preprocess + model)
    with open("schema.json") as f:
        schema = json.load(f)
    try:
        with open("metrics.json") as f:
            metrics = json.load(f)
    except Exception:
        metrics = None
    return model, schema, metrics

model, schema, metrics = load_artifacts()

st.title("👥 Employee Attrition Predictor")
st.caption("Uses your trained scikit-learn pipeline on HR tabular data")

# Show headline metrics if available
best_thr = 0.5
if metrics:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', np.nan):.3f}")
    with col2:
        st.metric("PR-AUC", f"{metrics.get('avg_precision', np.nan):.3f}")
    if "metrics_at_best_f1" in metrics and "threshold" in metrics["metrics_at_best_f1"]:
        best_thr = float(metrics["metrics_at_best_f1"]["threshold"])
        st.caption(f"Using decision threshold = {best_thr:.3f} (best F1 on validation)")
    st.divider()

st.subheader("Single Employee Input")

# Simple UI form from the saved schema
with st.form("input_form"):
    inputs = {}

    # Numeric
    for c in schema["num_cols"]:
        default = float(schema["defaults"].get(c, 0.0))
        step = 1.0 if float(default).is_integer() else 0.1
        inputs[c] = st.number_input(c, value=default, step=step)

    # Categorical
    for c in schema["cat_cols"]:
        options = schema["categories"].get(c, [])
        default = schema["defaults"].get(c, options[0] if options else "")
        if options and default not in options:
            default = options[0]
        idx = options.index(default) if options and default in options else 0
        inputs[c] = st.selectbox(c, options=options if options else [default], index=idx)

    submitted = st.form_submit_button("Predict Attrition Risk")

if submitted:
    expected_cols = schema["num_cols"] + schema["cat_cols"]
    X_one = pd.DataFrame([inputs], columns=expected_cols)
    proba = float(model.predict_proba(X_one)[:, 1][0])
    pred  = int(proba >= best_thr)

    st.success(f"Probability of Attrition (Yes): **{proba:.3f}**")
    st.write("Prediction:", "Yes (1)" if pred == 1 else "No (0)")

st.divider()
st.subheader("📤 Batch Predict (CSV)")

st.write("Upload a CSV with the **same columns** as your training features (after dropping ID columns).")
file = st.file_uploader("Upload CSV", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
    expected_cols = schema["num_cols"] + schema["cat_cols"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        Xb = df[expected_cols].copy()
        probs = model.predict_proba(Xb)[:, 1]
        preds = (probs >= best_thr).astype(int)
        out = df.copy()
        out["attrition_prob"] = probs
        out["attrition_pred"] = preds
        st.write(out.head())
        st.download_button("Download predictions CSV",
                           data=out.to_csv(index=False),
                           file_name="attrition_predictions.csv",
                           mime="text/csv")

st.caption("Tip: choose a lower threshold to catch more leavers (higher recall), or higher for fewer false alarms (higher precision).")
