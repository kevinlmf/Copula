# 🧠 Copula Modeling System

A dimension-adaptive statistical framework for modeling high-dimensional dependence using copulas. Inspired by Oh & Patton (2017), extended to modern applications involving sparse graphs, t-copulas, and Bayesian nonparametrics.

> 🔬 Ideal for financial time series, risk modeling, or any high-dimensional data where marginals and dependence need to be decoupled.

---

## 📦 Features

- **Dimension-Aware Model Selection**:
  - 🧊 Gaussian Copula for low dimensions (d ≤ 10)
  - 🧱 t-Factor Copula for medium dimensions (10 < d ≤ 50)
  - 🕸️ Graph-Regularized Factor Copula for high dimensions (50 < d ≤ 300)
  - 🌀 Dirichlet Process Mixture of Copulas for ultra-high dimensions (d > 300)

- **Statistical Evaluations**:
  - Kendall's tau matrix comparison
  - Tail dependence structure (coming soon)

- **Fully Modular Design**:
  - Each model is cleanly separated in its own file
  - Easy to plug in new copula families or estimators

---

## 🗂️ Project Structure

```
CopulaSystem/                            ← Root project folder
├── main.py                              ← Entry point: runs the pipeline
├── .gitignore                           ← Ignore venv, __pycache__, etc.
├── requirements.txt                     ← Python dependencies list
├── copula_modeling/                     ← Main codebase
│   ├── pipeline.py                      ← Dimension-adaptive copula controller
│   ├── data_preprocessing.py           ← Computes Kendall’s tau matrix
│   ├── evaluation.py                   ← Evaluation metrics and heatmaps
│   ├── utils.py                        ← General helper functions
│   └── models/                         ← Modeling components
│       ├── low_dim_models.py           ← Gaussian Copula (d ≤ 10)
│       ├── medium_dim_factor.py        ← Factor t-Copula (10 < d ≤ 50)
│       ├── high_dim_graph_factor.py    ← Graph-regularized model (50 < d ≤ 300)
│       └── infinite_dim_dpm.py         ← DPM-Factor Copula (d > 300)
```


---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/kevinlmf/Copula.git
cd Copula

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
📊 Sample Output

🚀 Starting Copula Modeling Pipeline...
📐 Data shape: n=300, d=20
✅ Using Factor t-Copula (medium-dim)
📏 Mean Kendall's tau error: 0.0813
📌 TODOs
 Implement EM or Variational Inference for t-Factor Copula

 Add Graphical Lasso estimator (high-dim)

 Integrate PyMC for DPM Copula inference

 Add unit tests

 Add time-varying copulas (optional)

📚 Reference
Oh & Patton (2017). Modeling Dependence in High Dimensions with Factor Copulas. Journal of Business & Economic Statistics.

⚖️ License
This project is open-source under the MIT License




