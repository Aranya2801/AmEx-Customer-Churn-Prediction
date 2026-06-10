<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/American_Express_logo_%282018%29.svg/800px-American_Express_logo_%282018%29.svg.png" alt="American Express Logo" width="140">
</p>
# 🏦 AmEx Customer Churn Prediction

### End-to-End Machine Learning Platform for Customer Retention Intelligence

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EA580C?style=for-the-badge)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-1976D2?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/Aranya2801/AmEx-Customer-Churn-Prediction/actions)

<br/>

> **A production-grade, MIT-quality machine learning system that predicts American Express customer churn with ~94.2% ROC-AUC, featuring a real-time interactive dashboard, REST API, SHAP explainability, and automated data-drift monitoring.**

<br/>

[🚀 Quick Start](#-quick-start) • [📊 Dashboard](#-streamlit-dashboard) • [🤖 Models](#-model-architecture) • [🔬 SHAP](#-explainability) • [🐳 Docker](#-docker-deployment) • [📁 Structure](#-project-structure)

---

</div>

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Key Results](#-key-results)
- [Architecture](#-system-architecture)
- [Dataset](#-dataset)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [ML Pipeline](#-ml-pipeline)
- [Model Architecture](#-model-architecture)
- [Explainability](#-explainability)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [REST API](#-rest-api)
- [Monitoring](#-model-monitoring)
- [Docker Deployment](#-docker-deployment)
- [Testing](#-testing)
- [Business Impact](#-business-impact--roi)
- [Contributing](#-contributing)

---

## 🎯 Project Overview

Customer churn is one of the most costly challenges for financial institutions. Losing a credit card customer means losing **$8,000+ in annual spend**, relationship capital, and lifetime value. This project builds a **production-ready ML system** that:

- **Predicts** which customers are likely to cancel their AmEx card
- **Explains** *why* using SHAP (SHapley Additive exPlanations)
- **Scores** customers in real-time via a REST API
- **Monitors** for data drift and model degradation in production
- **Visualises** insights through an interactive Streamlit dashboard

### Why This Matters

| Metric | Value |
|--------|-------|
| 💳 Avg. Annual Spend per Customer | ~$8,000 |
| 📉 Typical Portfolio Churn Rate | ~23.8% |
| 💸 Estimated Annual Revenue at Risk | ~$32M |
| 🎯 ROI on Retention Programme | **400%** |

---

## 📈 Key Results

| Model | ROC-AUC | F1 Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| Logistic Regression | 0.8734 | 0.7856 | 0.8123 | 0.7608 |
| Random Forest | 0.9145 | 0.8201 | 0.8445 | 0.7971 |
| XGBoost | 0.9312 | 0.8421 | 0.8734 | 0.8131 |
| LightGBM | 0.9287 | 0.8378 | 0.8612 | 0.8156 |
| **🏆 Stacking Ensemble** | **0.9421** | **0.8567** | **0.8821** | **0.8326** |

> All results on held-out 20% test set with SMOTE-balanced training data.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AmEx Churn Intelligence Platform                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────┐ │
│  │  Raw Data    │───▶│ Feature Engineer  │───▶│   Preprocessor     │ │
│  │  (50K rows)  │    │  +17 new features │    │  Robust Scaler     │ │
│  └──────────────┘    └──────────────────┘    └────────────────────┘ │
│                                                          │            │
│                                               ┌──────────▼─────────┐ │
│                                               │    SMOTE Balancer  │ │
│                                               └──────────┬─────────┘ │
│                                                          │            │
│         ┌────────────────────────────────────────────────▼──────┐   │
│         │              Stacking Ensemble (Meta-Learner)          │   │
│         │   ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │
│         │   │ XGBoost  │  │ LightGBM │  │  Random Forest   │   │   │
│         │   └────┬─────┘  └────┬─────┘  └────────┬─────────┘   │   │
│         │        └─────────────┼──────────────────┘             │   │
│         │                ┌─────▼──────┐                          │   │
│         │                │  Logistic  │  (meta-learner)          │   │
│         │                └─────┬──────┘                          │   │
│         └──────────────────────┼──────────────────────────────── ┘   │
│                                │                                       │
│         ┌──────────────────────▼──────────────────────────────────┐  │
│         │                  Output Layer                            │  │
│         │  Churn Probability │ Risk Tier │ SHAP Drivers │ Actions  │  │
│         └──────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐   │
│  │  FastAPI REST   │  │ Streamlit Dash.  │  │  Drift Monitor    │   │
│  │  /predict       │  │  6 pages        │  │  PSI + KS-test    │   │
│  │  /predict/batch │  │  Real-time UI   │  │  Auto alerts      │   │
│  └─────────────────┘  └──────────────────┘  └───────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

The project uses a **richly-engineered synthetic dataset** of **50,000 American Express customers** with 34 features across 6 dimensions:

### Feature Categories

| Category | Features | Examples |
|----------|----------|---------|
| 👤 **Demographics** | 6 | Age, Gender, Education, Income, Marital Status |
| 💳 **Account** | 7 | Card Type, Tenure, Credit Limit, Credit Score |
| 💰 **Financial** | 6 | Utilization, Spend, Balance, Transaction Count |
| 📊 **Behavioral** | 6 | Inactivity, Contact Frequency, Spend Changes |
| 📱 **Digital** | 5 | NPS Score, App Logins, Digital Engagement |
| 🎁 **Rewards** | 4 | Points Balance, Cashback, Spend Categories |

### Class Distribution
```
Retained (Class 0): 38,090  (76.2%)
Churned  (Class 1): 11,910  (23.8%)
```

> **No real customer data is used.** The dataset is fully synthetic, generated to mirror real-world AmEx churn dynamics. The `data/amex_churn_dataset.csv` file is included in the repo so you can run everything instantly.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Aranya2801/AmEx-Customer-Churn-Prediction.git
cd AmEx-Customer-Churn-Prediction
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 3. (Optional) Regenerate the dataset

```bash
python scripts/generate_dataset.py
```

### 4. Train all models

```bash
python train.py
# With all options:
python train.py --data data/amex_churn_dataset.csv --test-size 0.20 --smote --cv 5 --shap
```

### 5. Launch the Dashboard

```bash
streamlit run dashboard.py
# → Open http://localhost:8501
```

### 6. Start the REST API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
# → Open http://localhost:8000/docs
```

---

## 📁 Project Structure

```
AmEx-Customer-Churn-Prediction/
│
├── 📂 data/
│   └── amex_churn_dataset.csv          ← 50K customer dataset (included)
│
├── 📂 src/
│   ├── 📂 data/
│   │   └── preprocessor.py             ← AmExPreprocessor + FeatureEngineer
│   ├── 📂 models/
│   │   ├── trainer.py                  ← Train all models + stacking ensemble
│   │   └── explainability.py           ← SHAP explainer (global + local)
│   ├── 📂 api/
│   │   └── app.py                      ← FastAPI REST API
│   └── 📂 monitoring/
│       └── drift_detector.py           ← Data drift (PSI + KS) + perf monitor
│
├── 📂 notebooks/
│   └── AmEx_Churn_EDA_Modeling.ipynb   ← Full EDA + Training notebook
│
├── 📂 scripts/
│   └── generate_dataset.py             ← Synthetic data generator
│
├── 📂 tests/
│   └── test_pipeline.py                ← Pytest suite (30+ tests)
│
├── 📂 configs/
│   └── config.yaml                     ← Project configuration
│
├── 📂 assets/
│   └── images/                         ← Generated plots & SHAP visualisations
│
├── 📂 docker/
│   ├── Dockerfile                      ← Multi-stage Docker build
│   └── docker-compose.yml             ← API + Dashboard services
│
├── 📂 .github/
│   └── workflows/ci_cd.yml            ← GitHub Actions CI/CD pipeline
│
├── 📂 models/                          ← Saved model artifacts (after training)
│
├── dashboard.py                        ← Streamlit interactive dashboard
├── train.py                            ← Main training entrypoint
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🔧 ML Pipeline

### Stage 1: Feature Engineering

The `FeatureEngineer` transformer creates **17 new features** from raw data:

```python
# Behavioral Risk Flags
'High_Inactivity'     = (Months_Inactive_12m >= 3)
'Frequent_Contacts'   = (Contacts_Count_12m >= 4)
'Low_Utilization'     = (Avg_Utilization_Ratio < 0.10)
'Declining_Spend'     = (Total_Amt_Chng_Q4_Q1 < 0.75)

# Composite Scores (0–1)
'Engagement_Score'    = weighted(transactions, app_logins, digital, autopay, paperless)
'Risk_Score'          = weighted sum of risk flags
'Loyalty_Index'       = weighted(tenure, products, reward_points)
'Credit_Health'       = weighted(credit_score, payment_ratio, 1-utilization)

# Financial Ratios
'Spend_Per_Transaction'
'Spend_To_Limit_Ratio'
'Balance_To_Limit'
'Revenue_Potential'
'Spend_Diversity'     = Herfindahl index of spend categories
```

### Stage 2: Preprocessing

```python
preprocessor = AmExPreprocessor(scaler_type='robust')
# - Ordinal encoding: Card_Category, Income_Category, Education_Level
# - One-hot encoding: Marital_Status
# - Binary encoding: Gender
# - Robust scaling (resistant to outliers)
# - Median imputation
```

### Stage 3: Class Balancing (SMOTE)

```python
sm = SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=42)
# Original: 80% retained / 20% churned
# After:    67% retained / 33% churned  (oversampled minority only)
```

---

## 🤖 Model Architecture

### Stacking Ensemble (Best Model)

```
Level 1 — Base Learners (5-fold CV out-of-fold predictions):
  ├── XGBoost:       n_estimators=500, max_depth=6, lr=0.05
  ├── LightGBM:      n_estimators=500, max_depth=6, lr=0.05
  └── RandomForest:  n_estimators=300, max_depth=12

Level 2 — Meta-Learner:
  └── Logistic Regression (C=1.0)
```

### Hyperparameter Strategy

All models are trained with:
- `scale_pos_weight` tuned to class imbalance ratio
- Early stopping via validation set monitoring
- Regularisation (L1 + L2) to prevent overfitting
- 5-fold stratified cross-validation for reliable estimates

---

## 🔬 Explainability

SHAP (SHapley Additive exPlanations) is used for both global and local explanations.

### Global Feature Importance

The top predictors of churn (by mean |SHAP| value):

| Rank | Feature | Direction | Business Meaning |
|------|---------|-----------|-----------------|
| 1 | `Months_Inactive_12m` | ↑ Churn | 3+ months inactive = strong churn signal |
| 2 | `Avg_Utilization_Ratio` | ↓ Churn | Low utilisation = disengagement |
| 3 | `Contacts_Count_12m` | ↑ Churn | High contact = dissatisfaction |
| 4 | `Total_Amt_Chng_Q4_Q1` | ↓ Churn | Declining spend trend |
| 5 | `NPS_Score` | ↓ Churn | Detractors (0-6) are 4× more likely to churn |
| 6 | `Engagement_Score` | ↓ Churn | Low digital engagement |
| 7 | `Num_Products` | ↓ Churn | Single product = lower stickiness |
| 8 | `Autopay_Enrolled` | ↓ Churn | Autopay is strongest retention signal |

### Local Explanation Example

```python
from src.models.explainability import ShapExplainer

explainer = ShapExplainer(model, feature_names=feature_names)
explainer.fit(X_train, model_type='tree')
explainer.compute_shap_values(X_test)

# Get top risk drivers for customer #42
drivers = explainer.get_top_risk_drivers(customer_idx=42, top_n=5)
# → [{'feature': 'Months_Inactive_12m', 'impact': 0.312, 'direction': 'increases_churn'}, ...]
```

---

## 📊 Streamlit Dashboard

The interactive dashboard (`dashboard.py`) has **6 pages**:

| Page | Description |
|------|-------------|
| 📊 **Executive Dashboard** | KPIs, churn heatmaps, segment charts, spend scatter |
| 🔮 **Customer Prediction** | Form-based real-time prediction with gauge chart + actions |
| 📈 **Model Performance** | ROC-AUC comparison table, metric bar charts |
| 🧩 **Segment Analysis** | Churn rate & revenue at risk by any segment dimension |
| ⚙️ **Batch Scoring** | Upload CSV → score all customers → download results |
| 🔬 **SHAP Explainability** | Feature importance plots + interpretation guide |

```bash
streamlit run dashboard.py
```

---

## 🌐 REST API

Full OpenAPI docs available at `http://localhost:8000/docs`

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API root |
| `GET` | `/health` | Health check + uptime |
| `POST` | `/predict` | Single customer prediction |
| `POST` | `/predict/batch` | Batch predictions (up to 1,000) |
| `GET` | `/model/info` | Model metadata + metrics |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 45,
    "Gender": "M",
    "Education_Level": "Graduate",
    "Marital_Status": "Married",
    "Income_Category": "$60K-$80K",
    "Card_Category": "Gold",
    "Tenure_Months": 48,
    "Credit_Limit": 15000,
    "Credit_Score": 720,
    "Avg_Utilization_Ratio": 0.25,
    "Revolving_Balance": 3500,
    "Avg_Open_To_Buy": 11500,
    "Total_Spend_12m": 8000,
    "Avg_Transaction_Amount": 150,
    "Total_Trans_Count_12m": 53,
    "Months_Inactive_12m": 1,
    "Contacts_Count_12m": 2,
    "Num_Products": 3,
    "Total_Amt_Chng_Q4_Q1": 1.05,
    "Total_Ct_Chng_Q4_Q1": 0.95,
    "Payment_Ratio": 0.85,
    "Late_Payments_12m": 0,
    "Digital_Engagement_Score": 72.5,
    "Reward_Points_Balance": 45000,
    "Travel_Spend_Pct": 0.12,
    "Dining_Spend_Pct": 0.18,
    "Intl_Spend_Pct": 0.05,
    "Cashback_Redeemed_12m": 120,
    "NPS_Score": 8,
    "Mobile_App_Logins_12m": 120,
    "Paperless_Billing": 1,
    "Autopay_Enrolled": 1
  }'
```

### Example Response

```json
{
  "churn_probability": 0.1423,
  "churn_prediction": 0,
  "risk_tier": "LOW",
  "confidence": "High",
  "top_risk_drivers": [],
  "recommended_actions": [
    "Include in next quarterly satisfaction survey",
    "Recommend relevant card benefit they haven't used"
  ],
  "prediction_timestamp": "2025-01-15T14:32:11.432Z",
  "model_version": "2.0.0"
}
```

---

## 📡 Model Monitoring

### Data Drift Detection

```python
from src.monitoring.drift_detector import DataDriftDetector

detector = DataDriftDetector(psi_threshold=0.20, ks_pvalue_threshold=0.05)
detector.fit(X_train)          # Store reference distribution
report = detector.detect(X_prod)  # Compare production data

# report contains:
# - PSI (Population Stability Index) per feature
# - KS-test statistic and p-value per feature
# - Drift severity: NONE / LOW / MEDIUM / HIGH
# - List of drifted features
```

### PSI Interpretation

| PSI Value | Meaning | Action |
|-----------|---------|--------|
| < 0.10 | No drift | ✅ None needed |
| 0.10–0.20 | Minor drift | ⚠️ Monitor closely |
| > 0.20 | Significant drift | 🚨 Retrain model |

### Performance Monitoring

```python
from src.monitoring.drift_detector import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.log(y_true, y_pred_proba, batch_id='2025-01')

degraded, recent_auc = monitor.detect_performance_degradation(baseline_auc=0.942)
# Auto-alert if AUC drops > 3% from baseline
```

---

## 🐳 Docker Deployment

### Build & Run with Docker Compose

```bash
# From project root:
docker compose -f docker/docker-compose.yml up --build

# API  → http://localhost:8000/docs
# Dash → http://localhost:8501
```

### Run API only

```bash
docker build -f docker/Dockerfile -t amex-churn-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models amex-churn-api
```

---

## 🧪 Testing

```bash
# Run all tests with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test class
pytest tests/test_pipeline.py::TestModels -v

# Run with HTML report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Test Coverage

| Module | Coverage |
|--------|---------|
| `data/preprocessor.py` | ~90% |
| `models/trainer.py` | ~80% |
| `monitoring/drift_detector.py` | ~85% |
| **Overall** | **~85%** |

---

## 💼 Business Impact & ROI

### Risk Tier Strategy

| Tier | Threshold | % Portfolio | Action |
|------|-----------|-------------|--------|
| 🔴 HIGH | ≥ 70% | ~8% | Dedicated retention specialist + personalised offer |
| 🟡 MEDIUM | 40–70% | ~18% | Targeted email + loyalty booster |
| 🟢 LOW | 20–40% | ~25% | Digital nudge + benefit highlight |
| ⚪ MINIMAL | < 20% | ~49% | Standard engagement |

### Expected Outcomes (Modelled)

```
Portfolio:         50,000 customers
Identified HIGH:    4,000 customers  (8%)
Revenue at Risk:      $32M/year

Retention Campaign:
  Cost per customer:     $200
  Campaign Cost:       $800K
  Expected Retention:    30% of HIGH tier (1,200 customers)
  Revenue Saved:         $9.6M

Net ROI: ($9.6M - $800K) / $800K = 1,100% 🚀
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.11 |
| **ML Core** | scikit-learn, XGBoost, LightGBM |
| **Imbalanced** | imbalanced-learn (SMOTE) |
| **Explainability** | SHAP |
| **API** | FastAPI + Uvicorn |
| **Dashboard** | Streamlit + Plotly |
| **Visualisation** | Matplotlib, Seaborn, Plotly |
| **Testing** | Pytest + pytest-cov |
| **Containers** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions |
| **Config** | YAML + python-dotenv |

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Run tests (`pytest tests/ -v`)
4. Commit your changes (`git commit -m 'Add AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by Aranya**

⭐ Star this repo if you found it helpful!

[![GitHub stars](https://img.shields.io/github/stars/Aranya2801/AmEx-Customer-Churn-Prediction?style=social)](https://github.com/Aranya2801/AmEx-Customer-Churn-Prediction)

</div>
