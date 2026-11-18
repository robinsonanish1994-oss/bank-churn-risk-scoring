# Predictive Modeling and Risk Scoring for Bank Customer Churn

This project builds a churn intelligence system for 10,000 European retail-bank customers. It trains and compares five models, explains their predictions with SHAP, and serves the results through a Streamlit dashboard.

## Results at a glance
| | |
|---|---|
| Best model | Gradient Boosting: test ROC-AUC **0.870**, 5-fold CV AUC 0.862 ± 0.008 |
| Runner-up | XGBoost: test ROC-AUC 0.869 |
| Lift | Top 10% riskiest customers: 86% churn, 42% of all churners |
| Top drivers | Age, number of products, Germany, gender, engagement × products, activity |

## Structure
```
app.py                     Streamlit dashboard (7 modules)
src/features.py            Shared preprocessing + feature engineering
src/train.py               EDA → training → evaluation → SHAP/PDP → scoring
data/European_Bank.csv     Raw data
data/scored_customers.csv  Every customer with probability, risk band, flag
models/best_model.joblib   Gradient Boosting + decision thresholds
models/logistic_regression.joblib
reports/metrics.json       All metrics, EDA tables, importances
reports/figures/           14 figures (EDA, ROC/PR, confusion, lift, SHAP, PDP)
```

## Run
```bash
pip install -r requirements.txt
python src/train.py        # optional: regenerates models/ and reports/ (~1 min)
streamlit run app.py
```

## Dashboard modules
1. **Portfolio overview**: KPIs, risk bands, and churn by market × engagement and by age
2. **Risk calculator**: enter a customer's details to get a probability gauge, churn flag, SHAP "why this score" chart and suggested actions
3. **What-if simulator**: change activity, products, card and balance and compare the baseline and scenario scores, with age and balance sweeps
4. **Probability distribution**: score separation, an interactive threshold with live precision/recall, and a decile lift chart
5. **Feature importance**: SHAP ranking, logistic-regression coefficients, beeswarm and partial dependence plots
6. **Model performance**: comparison table, threshold policies, ROC curves and confusion matrices
7. **Customer lookup**: filter by risk band, market and engagement, then export a CSV target list

## Deploy (Streamlit Community Cloud)
Push this folder to GitHub, then create a new app on share.streamlit.io with `app.py` as the main file.
