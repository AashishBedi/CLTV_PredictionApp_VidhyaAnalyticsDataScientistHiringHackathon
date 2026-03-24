# 🛡️ VahanBima — Customer Lifetime Value (CLTV) Predictor

> **AI-powered CLTV prediction platform** for VahanBima insurance customers — segment customers intelligently and unlock personalized insurance journeys.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

This project was built for the **Analytics Vidhya Data Scientist Hiring Hackathon**. The goal is to predict the **Customer Lifetime Value (CLTV)** of VahanBima auto-insurance customers and segment them into actionable tiers — enabling relationship managers to prioritize retention, upsell, and engagement strategies.

---

## 🏆 Customer Tiers

| Tier | CLTV Range | Strategy |
|------|-----------|----------|
| 💎 Platinum | ≥ ₹1,50,000 | Elite retention & VIP perks |
| ⭐ Gold | ₹80,000 – ₹1,50,000 | Upsell & loyalty rewards |
| 🏅 Silver | < ₹80,000 | Cross-sell & engagement campaigns |

---

## ✨ Features

- 🔮 **CLTV Prediction** — Predict lifetime value from 10 customer inputs
- 🗂️ **Customer Segmentation** — Automatic Platinum / Gold / Silver classification
- 📊 **CLTV Spectrum Chart** — Visual gauge showing where the customer lands across tiers
- ⚠️ **Risk Flagging** — Highlights high claim-to-CLTV ratio profiles automatically
- 🎨 **Modern Dark UI** — Glassmorphism design with gradient styling

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/AashishBedi/CLTV_PredictionApp_VidhyaAnalyticsDataScientistHiringHackathon.git
cd CLTV_PredictionApp_VidhyaAnalyticsDataScientistHiringHackathon

# 2. Install dependencies
pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn

# 3. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 🖥️ Usage

1. Fill in the **Customer Profile** in the sidebar (demographics, income, marital status)
2. Fill in **Policy Details** (policy type, tier, number of policies)
3. Adjust **Claim Information** (vintage years, claim amount)
4. Click **🔮 Predict Customer CLTV**
5. View the predicted CLTV, tier classification, claim ratio, and the CLTV spectrum chart

---

## ⚙️ Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest Regressor |
| Estimators | 200 trees |
| Training Records | 89,392 |
| Engineered Features | 14 |
| R² Score | 0.1521 |

### 🧠 Engineered Features

| Feature | Description |
|---------|------------|
| `income_ord` | Ordinal encoding of income band |
| `policy_tier` | Ordinal encoding of Silver/Gold/Platinum |
| `multi_policy` | Binary flag for multiple policies |
| `is_urban` | Binary flag for urban area |
| `is_male` | Binary gender flag |
| `claim_per_year` | Claim amount normalized by vintage |
| `income_x_vintage` | Income × vintage interaction |
| `tier_x_claim` | Policy tier × claim amount interaction |
| `vintage_x_claim` | Vintage × claim amount interaction |

---

## 📁 Project Structure

```
VahanBima_CLTV/
│
├── app.py                    # 🎨 Streamlit frontend app
├── CLTV_pipeline.ipynb       # 📓 Model training & EDA notebook
├── models/
│   └── rf_cltv_model.pkl     # 🤖 Trained Random Forest model
│
├── train_file.csv            # 📦 Training dataset
├── test_file.csv             # 📦 Test dataset
├── sample_submission.csv     # 📄 Submission template
├── submission.csv            # ✅ Final predictions
│
├── cltv_distribution.png     # 📊 CLTV distribution plot
├── correlation_heatmap.png   # 📊 Feature correlation heatmap
├── categorical_vs_cltv.png   # 📊 Categorical features vs CLTV
└── numerical_vs_cltv.png     # 📊 Numerical features vs CLTV
```

---

## 📊 EDA Highlights

- CLTV is **right-skewed** with most customers concentrated in the Silver tier
- `vintage`, `claim_amount`, and `income` are the strongest predictors
- Platinum customers represent ~12% of the base but disproportionately high revenue

---

## 🙋‍♂️ Author

**Aashish Bedi**
- Hackathon: Analytics Vidhya — Data Scientist Hiring Challenge
- Platform: [Analytics Vidhya](https://www.analyticsvidhya.com/)

---
