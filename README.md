# Metaheuristic IDS Optimization (AIM Coursework)

This project evaluates an Intrusion Detection System (IDS) pipeline using a baseline Random Forest and six metaheuristic variants. The optimizers perform joint feature selection alongside hyperparameter tuning (`n_estimators`, `max_depth`, `min_samples_split`) on the CICIDS2017 network traffic dataset.

### Evaluated Models
- Baseline Random Forest (all features)
- Genetic Algorithm (GA) + Random Forest
- Particle Swarm Optimization (PSO) + Random Forest
- Grey Wolf Optimizer (GWO) + Random Forest
- Adaptive GWO + Random Forest
- GA-PSO Hybrid + Random Forest
- NSGA-III + Random Forest

## Key Findings
- **Highest Accuracy & Balanced Recall:** NSGA-III + RF provides the strongest multi-objective balance, yielding high attack recall with a compact feature set, making it highly viable for general SOC deployment.
- **Strict False Positive Minimization:** Baseline Random Forest consistently suppresses the highest number of false alarms, functioning best in environments aiming to avoid alert fatigue.
- **Fastest Computational Runtime:** PSO + RF offers the lowest latency for both training and inference.

## Dataset

The project uses standard CICIDS2017 CSV files located under:
- `data/MachineLearningCVE/`

**Pipeline Execution:**
- The workflow concatenates the full cleaned data first, standardizing column headers.
- It randomly samples 15% from the full concatenated dataset (reproducible via a fixed `random_state=42`).
- It extracts a robust stratified train/test split.
- Compiled split files are saved to `data/combined_ml_15pct/` for persistence across optimization runs.

## Project Structure

- `Eval.ipynb`: Main experiment notebook containing the full execution pipeline and visualizations.
- `requirements.txt`: Python package dependencies.
- `preprocessing/clean.py`: Data loading, standardization, and stratified splitting logic.
- `core/baseline.py`: Baseline model training and performance evaluation.
- `core/fitness.py`: Optimization objective function and final model evaluation metrics.
- `optimizers/ga.py`: Genetic Algorithm (GA).
- `optimizers/pso.py`: Particle Swarm Optimization (PSO).
- `optimizers/gwo.py`: Grey Wolf Optimizer (GWO).
- `optimizers/adaptive_gwo.py`: Dynamic parameter Adaptive GWO.
- `optimizers/gapso.py`: Hybridized GA and PSO strategy.
- `optimizers/nsga3.py`: Non-dominated Sorting Genetic Algorithm III (NSGA-III).

## Installation

1) Create and activate a virtual environment (recommended):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:
```powershell
pip install -r requirements.txt
```

## How To Run

**Option A: Notebook (Recommended)**
1. Open `Eval.ipynb`.
2. Run the cells from top to bottom. The notebook will automatically build/load the sampled split files under `data/combined_ml_15pct/`.
3. Review the metrics table, feature allocations, and visualizations.

**Provided Evaluation Outputs:**
- Classification metrics (Accuracy, Precision, Recall, F1)
- Operations errors (False Positives and False Negatives counts)
- Latency (Training and Testing time comparisons)
- Feature-selection overlap and dimensional importance.
- Normalized optimizer convergence trajectory graphing.

## Reproducibility Tips
- Optimization settings (agents/iterations) are highly configurable in `Eval.ipynb`. Higher iterations improve convergence tracking but noticeably increase compute overhead.
- Keep `random_state` fixed where possible to strictly mirror the paper's reported outputs.
- Run all algorithm comparisons sequentially on the exact same dataset split to prevent sample-variance skew.

## Troubleshooting

### CSV Loads But 'label' Column Is Missing
If you see an error ending in: `Error: Target column 'label' not found.` or `ValueError: not enough values to unpack`, your dataset files are likely Git LFS pointer files instead of the real CSV data.

**Fix:**
Run the following inside your terminal to pull the raw binary sets:
```powershell
git lfs pull origin main
```

## Authors
**AIM Coursework Project**
University of Nottingham Malaysia
- Rajul Kabir 
- Imitaz Naufal 
- Babacar Sene
