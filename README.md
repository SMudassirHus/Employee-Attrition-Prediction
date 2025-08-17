# Employee Attrition Prediction (Tabular ML)

End-to-end ML project to predict if an employee will leave (Attrition = Yes) using the IBM HR dataset.

## What’s inside
- `notebook.ipynb` – training, evaluation, permutation importance
- `best_attrition_model.joblib` – trained pipeline (preprocess + model)
- `schema.json` – UI schema for the Streamlit app
- `metrics.json` – ROC-AUC, PR-AUC, F1 @ tuned threshold
- `app.py` – Streamlit app (single prediction + batch CSV)
- `requirements.txt` – Python deps
- `assets/` – plots (top features, ROC, PR, confusion matrix)

## Try the App
Deployed on Hugging Face Spaces: *(https://huggingface.co/spaces/Mudassir110/employee-attrition-app1)*

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
