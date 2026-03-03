# ExplainAI — Model Transparency Visualizer

> **MSc Computer Science & Artificial Intelligence**  
> A production-ready full-stack XAI system implementing SHAP, LIME, and Feature Importance explanations for scikit-learn compatible machine learning models.

---

## Table of Contents

1. [Project Architecture](#1-project-architecture)
2. [Academic Foundations](#2-academic-foundations)
3. [Folder Structure](#3-folder-structure)
4. [Local Development Setup](#4-local-development-setup)
5. [Docker Deployment](#5-docker-deployment)
6. [Production Deployment](#6-production-deployment)
7. [API Reference](#7-api-reference)
8. [Generating Test Data](#8-generating-test-data)
9. [Future Research Extensions](#9-future-research-extensions)

---

## 1. Project Architecture

```
Browser (React + Plotly.js)
        │
        │  HTTP/JSON  (X-Session-ID header)
        ▼
FastAPI Backend (Python)
   ├── POST /api/upload-model    → model_handler.py (joblib + allowlist)
   ├── POST /api/upload-data     → data_validator.py (CSV + compat check)
   ├── POST /api/generate-explanations
   │       ├── SHAP (TreeExplainer / LinearExplainer / KernelExplainer)
   │       └── LIME (LimeTabularExplainer)
   └── GET  /api/metrics         → metrics_service.py
```

**Session model:** The backend uses a lightweight in-memory session store keyed by `X-Session-ID`. The frontend receives the session ID on model upload and sends it in all subsequent requests. **In production with multiple workers, replace with Redis.**

---

## 2. Academic Foundations

### 2.1 SHAP — Global & Local Interpretability

SHAP (SHapley Additive exPlanations) decomposes every prediction as:

```
f(x) = φ₀ + φ₁ + φ₂ + ... + φₘ
```

Where:
- `φ₀ = E[f(X)]` — the base value (mean model output)
- `φᵢ` — the Shapley value for feature `i` (signed contribution)

The Shapley values are the **unique solution** satisfying:
- **Efficiency:** Σφᵢ = f(x) − φ₀ (values sum to the prediction gap)
- **Symmetry:** Equal contributors get equal values
- **Dummy:** Unused features get φᵢ = 0
- **Linearity:** Additive decomposition across features

**TreeExplainer** computes exact Shapley values for tree ensembles in O(TLD²) — polynomial time. This makes it practical for RandomForest, XGBoost, and GradientBoosting.

**Global interpretability:** Aggregate mean |φᵢ| across all N instances.  
**Local interpretability:** Individual φᵢ for one specific prediction.

> Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017.  
> Reference: Lundberg et al., "From Local Explanations to Global Understanding with Explainable AI for Trees," Nature Machine Intelligence, 2020.

### 2.2 LIME — Local Surrogate Models

LIME constructs a locally faithful linear surrogate around instance x*:

```
Step 1: Sample N perturbed instances x'ᵢ near x*
Step 2: Get black-box predictions: f(x'₁), ..., f(x'ₙ)
Step 3: Weight by proximity: πᵢ = exp(-‖x* - x'ᵢ‖² / σ²)
Step 4: Fit weighted ridge regression: g = argmin Σ πᵢ(f(x'ᵢ) - g(x'ᵢ))² + Ω(g)
Step 5: Return top-K coefficients of g
```

**Why LIME is always LOCAL:** The surrogate `g` is trained only in the neighbourhood of `x*`. A different instance produces a completely different surrogate — this correctly reflects the non-linearity of the black-box model.

> Reference: Ribeiro, Singh, Guestrin, "Why Should I Trust You?: Explaining the Predictions of Any Classifier," KDD 2016.

### 2.3 Global vs. Local Interpretability

| Dimension | Global | Local |
|-----------|--------|-------|
| **Scope** | Whole dataset | Single prediction |
| **Question** | Which features matter overall? | Why this specific prediction? |
| **SHAP** | Mean \|SHAP\| bar chart, beeswarm | Force plot for instance x* |
| **LIME** | Not applicable | Local linear surrogate |
| **Use case** | Model auditing, feature selection | Debugging individual decisions |

---

## 3. Folder Structure

```
explainai/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app + middleware + routers
│   │   ├── routes/
│   │   │   ├── model_routes.py        # POST /upload-model
│   │   │   ├── data_routes.py         # POST /upload-data
│   │   │   ├── explanation_routes.py  # POST /generate-explanations
│   │   │   └── metrics_routes.py      # GET /metrics
│   │   ├── services/
│   │   │   ├── model_handler.py       # Model loading + validation
│   │   │   ├── explanation_engine.py  # SHAP + LIME core
│   │   │   ├── data_validator.py      # CSV ingestion + preprocessing
│   │   │   └── metrics_service.py     # Performance metrics
│   │   ├── schemas/
│   │   │   └── schemas.py             # Pydantic v2 request/response models
│   │   └── utils/
│   │       └── session_store.py       # Thread-safe in-memory sessions
│   ├── generate_test_data.py          # Demo model + dataset generator
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── main.jsx                   # React entry point
│   │   ├── App.jsx                    # Root component + tab routing
│   │   ├── index.css                  # Tailwind + global styles
│   │   ├── hooks/
│   │   │   └── useSession.jsx         # Session context + API client
│   │   └── components/
│   │       ├── Header.jsx             # Status badges + branding
│   │       ├── UploadPanel.jsx        # Model & data upload UI
│   │       ├── ExplanationDashboard.jsx # SHAP + LIME visualizations
│   │       ├── MetricsPanel.jsx       # Performance metrics + confusion matrix
│   │       └── Footer.jsx             # References + stack info
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── vercel.json
│   └── index.html
│
└── docker-compose.yml
```

---

## 4. Local Development Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### Backend Setup

```bash
# 1. Navigate to backend
cd explainai/backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the FastAPI server
uvicorn app.main:app --reload --port 8000

# API will be available at:
# http://localhost:8000
# http://localhost:8000/docs  ← Swagger UI
# http://localhost:8000/redoc ← ReDoc
```

### Frontend Setup

```bash
# 1. Navigate to frontend
cd explainai/frontend

# 2. Install dependencies
npm install

# 3. Configure API URL
cp .env.example .env.local
# Edit .env.local: VITE_API_URL=http://localhost:8000/api

# 4. Start Vite dev server
npm run dev

# App will be available at http://localhost:5173
```

### Generate Test Data

```bash
# In backend directory with venv activated
cd explainai/backend
python generate_test_data.py

# Creates:
#   test_model.pkl   (RandomForestClassifier, 12 features)
#   test_data.csv    (150 rows, target column = "label")
```

---

## 5. Docker Deployment

### Build and run with Docker Compose (recommended)

```bash
# From project root
cd explainai

# Build and start backend
docker-compose up --build

# Backend available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Manual Docker build

```bash
cd explainai/backend

# Build image
docker build -t explainai-backend:latest .

# Run container
docker run -p 8000:8000 \
  -e ALLOWED_ORIGINS="http://localhost:3000,https://your-frontend.vercel.app" \
  -v $(pwd)/uploads:/home/explainai/app/uploads \
  explainai-backend:latest
```

---

## 6. Production Deployment

### 6.1 Backend → Render.com

1. **Push to GitHub:**
   ```bash
   git init && git add . && git commit -m "initial"
   git remote add origin https://github.com/yourusername/explainai.git
   git push
   ```

2. **Create Render Web Service:**
   - Go to [render.com](https://render.com) → New → Web Service
   - Connect GitHub repository
   - Set Root Directory: `backend`
   - Runtime: **Docker**
   - Dockerfile path: `./Dockerfile`

3. **Environment Variables on Render:**
   ```
   ALLOWED_ORIGINS = https://your-explainai.vercel.app
   ```

4. **Note the backend URL** (e.g., `https://explainai-backend.onrender.com`)

### 6.2 Backend → AWS EC2

```bash
# On EC2 instance (Amazon Linux 2 / Ubuntu)
sudo yum install docker -y           # or: sudo apt install docker.io -y
sudo systemctl start docker
sudo usermod -aG docker ec2-user

# Clone and build
git clone https://github.com/yourusername/explainai.git
cd explainai/backend
docker build -t explainai-backend .

# Run with nginx reverse proxy (recommended)
docker run -d -p 8000:8000 \
  -e ALLOWED_ORIGINS="https://your-frontend.vercel.app" \
  --restart=always \
  --name explainai \
  explainai-backend

# Configure Security Group: allow port 80, 443, 8000
```

### 6.3 Frontend → Vercel

```bash
# Install Vercel CLI
npm i -g vercel

cd explainai/frontend

# Add environment variable
echo "VITE_API_URL=https://your-backend.onrender.com/api" > .env.production

# Deploy
vercel --prod

# Or connect GitHub to Vercel for auto-deployments
```

**In Vercel Dashboard:** Set Environment Variable:
```
VITE_API_URL = https://your-explainai-backend.onrender.com/api
```

---

## 7. API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload-model` | Upload `.pkl` model file |
| `POST` | `/api/upload-data` | Upload `.csv` dataset |
| `POST` | `/api/generate-explanations` | Run SHAP + LIME pipeline |
| `GET`  | `/api/metrics` | Compute performance metrics |
| `GET`  | `/health` | Liveness probe |

### Headers (all requests after model upload)
```
X-Session-ID: <uuid>   # Returned from upload-model response
```

### POST /api/generate-explanations — Body
```json
{
  "instance_index": 0,
  "num_lime_features": 10,
  "num_lime_samples": 5000,
  "background_sample_size": 100
}
```

Full interactive documentation: `http://localhost:8000/docs`

---

## 8. Generating Test Data

```bash
cd backend
python generate_test_data.py
```

Upload sequence in the UI:
1. Upload `test_model.pkl`
2. Upload `test_data.csv`, set target column = `label`
3. Run analysis, instance index = `0`

---

## 9. Future Research Extensions

### 9.1 Counterfactual Explanations
Answer: *"What is the minimum change to this instance that would flip the prediction?"*

**Recommended library:** [DiCE (Diverse Counterfactual Explanations)](https://github.com/interpretml/DiCE)
```python
# Implementation sketch
import dice_ml
data = dice_ml.Data(dataframe=df, continuous_features=features, outcome_name="label")
model_d = dice_ml.Model(model=clf, backend="sklearn")
exp = dice_ml.Dice(data, model_d)
cf = exp.generate_counterfactuals(query_instance, total_CFs=4, desired_class="opposite")
```

### 9.2 Fairness & Bias Detection
Audit model decisions for protected attributes (race, gender, age).

**Recommended libraries:**
- [IBM AI Fairness 360 (AIF360)](https://aif360.mybluemix.net/)
- [Fairlearn](https://fairlearn.org/)

Key metrics: **Demographic Parity Difference**, **Equalized Odds**, **Individual Fairness**

```python
# Implementation sketch
from aif360.metrics import BinaryLabelDatasetMetric
metric = BinaryLabelDatasetMetric(dataset, 
    unprivileged_groups=[{"race": 0}], 
    privileged_groups=[{"race": 1}])
print(metric.disparate_impact())
```

### 9.3 Model Comparison Module
Side-by-side XAI comparison for multiple models on the same dataset.

**Architecture extension:**
- Upload multiple models per session
- Compute SHAP for each
- Rank feature agreement (Spearman correlation of feature rankings)
- Detect prediction disagreements at the instance level

### 9.4 Concept-Based Explanations (TCAV)
For deep learning models: identify which human-defined concepts a model has learned.

**Reference:** Kim et al., "Interpretability Beyond Classification," ICML 2018.

### 9.5 Attention Visualization
For transformer-based models (BERT, ViT): visualize multi-head attention matrices as a proxy for feature importance.

---

## Academic Note on Explanation Faithfulness

A critical open research question: **Are post-hoc explanations faithful to the model?**

- SHAP satisfies formal axiomatic guarantees (Shapley axioms)
- LIME's surrogate may be locally unfaithful in high-curvature regions
- Both are **post-hoc** — they explain the model's output, not its internal mechanism

For high-stakes decisions (medical, legal), consider **inherently interpretable models** (GAMs, decision trees) over post-hoc explanations of black-boxes.

> Rudin, C., "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead," Nature Machine Intelligence, 2019.
