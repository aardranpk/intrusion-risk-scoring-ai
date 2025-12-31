# 🔐 Explainable Intrusion Risk Scoring System (CICIDS)

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-GradientBoosting-red)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Security](https://img.shields.io/badge/Cybersecurity-Intrusion%20Detection-darkgreen)
[![Author](https://img.shields.io/badge/Author-Aardran%20Premakumar-blueviolet)](https://github.com/<your-username>)


An **end-to-end, explainable AI system** for network intrusion detection that goes beyond binary alerts by producing **risk scores, severity levels, and human-interpretable explanations** for security analysts.

This project demonstrates how machine learning can support **real-world SOC (Security Operations Center) decision-making**, not just model accuracy.

---

## 🚨 Problem Statement

Traditional intrusion detection systems often generate **binary alerts** (attack / no attack), which leads to:

- Alert fatigue for security analysts  
- Poor prioritization of threats  
- Lack of transparency in ML-driven decisions  

In real security operations, analysts need answers to:
- *How risky is this event?*
- *Why was it flagged?*
- *What should I investigate first?*

---

## 🎯 Solution Overview

This project implements an **Explainable Intrusion Risk Scoring System** that:

- Detects malicious network traffic using **XGBoost**
- Outputs a **risk score (0–100)** instead of a binary label
- Assigns **severity levels** (Low / Medium / High / Critical)
- Explains predictions using **SHAP feature attributions**
- Provides an **interactive Streamlit dashboard** for analysts

---

## 🧠 Key Features

- Risk-based scoring rather than yes/no predictions  
- Explainable AI (XAI) using SHAP for transparency  
- Imbalance-aware modeling for realistic attack detection  
- Analyst-friendly dashboard for exploration and triage  
- Production-style project structure (no notebooks required)

---

## 🏗️ Architecture

The system follows a production-style ML pipeline designed for SOC workflows:

Network Flow Data (CICIDS)
<br>│
<br>▼
<br>Data Loader & Preprocessing
<br>│
<br>▼
<br>XGBoost Classifier
<br>│
<br>├── Probability (0–1)
<br>│
<br>├── Risk Score (0–100)
<br>│ └── Severity Mapping
<br>│
<br>└── SHAP Explainer
<br>│
<br>▼
<br>Analyst Dashboard (Streamlit)


---

## 📊 Dataset

- **CICIDS 2017** (Canadian Institute for Cybersecurity)
- Network flow-level features
- Includes benign traffic and multiple attack scenarios  
- Dataset files are **not committed** (handled via `.gitignore`)

---

## 🛠️ Tech Stack

- Python 3
- Pandas, NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Joblib

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/aardranpk/intrusion-risk-scoring-ai.git
cd intrusion-risk-scoring-ai

```
### 2️⃣Create & activate virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1

```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt

```
## 📁 Data Setup

Create a data/ folder in the project root and place one CICIDS parquet file inside:

data/
 └── Portscan-Friday-no-metadata.parquet
Dataset files are excluded from version control.

## 🚀 Running the Project

#### Train baseline model
```bash
python -m src.models.train_baseline

```

#### Train XGBoost risk scoring model
```bash
python -m src.models.train_xgb

```

#### Run risk scoring demo
```bash
python -m src.models.demo_risk_score

```

#### Run SHAP explanation demo
```bash
python -m src.explain.demo_explain


```

#### Launch dashboard
```bash
python -m streamlit run dashboard/app.py

```

## 📈 Example Results

- ROC-AUC ≈ **0.999**
- High recall for attack traffic
- Risk score + severity per network flow
- Top contributing features for every prediction

---

## 🔍 Explainability Example (SHAP)

Fwd Packet Length Mean → ↓ risk
<br>Init Bwd Win Bytes → ↑ risk
<br>Total Backward Packets → ↓ risk


This allows analysts to **understand and trust** model decisions.

---

## ⚠️ Limitations

- Single dataset source (CICIDS)
- No real-time streaming ingestion
- No analyst feedback loop

---

## 🔮 Future Improvements

- Multi-class attack classification
- Real-time log ingestion (streaming)
- Analyst feedback loop for retraining
- Model drift detection
- Authentication & RBAC for dashboard

---

## 💼 Resume Highlight

> Built an explainable AI-based intrusion risk scoring system using XGBoost and SHAP, enabling risk-based prioritization of network traffic with transparent, analyst-facing explanations.

---

## 📌 Why This Project Matters

This project focuses on **how AI is used in security operations**, not just model accuracy.  
It demonstrates:

- Systems thinking
- Explainable ML
- Practical security trade-offs
- Production-ready engineering practices

---

## 👤 Author

**Aardran Premakumar**  
AI / ML | Security Analytics
