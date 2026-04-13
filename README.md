# Metaheuristic IDS Optimization (AIM Coursework)

This project evaluates an Intrusion Detection System (IDS) pipeline using:
- Baseline Random Forest (all features)
- Genetic Algorithm (GA) + Random Forest
- Particle Swarm Optimization (PSO) + Random Forest
- Grey Wolf Optimizer (GWO) + Random Forest

The optimizers perform joint feature selection and hyperparameter tuning on CICIDS2017 traffic data.

## Project Goals

- Build a reproducible IDS evaluation workflow in Python
- Compare baseline vs optimized models using detection quality and efficiency metrics
- Analyze feature reduction impact and optimizer behavior over iterations

## Dataset

The project currently uses CICIDS2017 CSV files under:
- data/MachineLearningCVE/

Current evaluation mode:
- Uses all CSVs in data/MachineLearningCVE/
- Concatenates full cleaned data first
- Randomly samples 15% from the full concatenated set
- Creates a stratified train/test split
- Saves split files to data/combined_ml_15pct/

How the 15% is computed:
- Let N be the total number of cleaned rows after all MachineLearningCVE CSVs are concatenated
- The sampled dataset size is 0.15 x N (global sample, not 15% per file)
- With fixed random_state=42, the sampled rows are reproducible across runs unless force_rebuild is enabled
- The sampled dataset is then split into train/test using stratification on the binary label

## Project Structure

- Eval.ipynb: Main experiment notebook (full analysis + plots)
- requirements.txt: Python dependencies
- preprocessing/clean.py: Data loading and preprocessing
- core/baseline.py: Baseline model training/evaluation
- core/fitness.py: Optimization objective and final model evaluation
- optimizers/ga.py: Genetic Algorithm optimizer
- optimizers/pso.py: Particle Swarm optimizer
- optimizers/gwo.py: Grey Wolf optimizer

## Installation

1) Create and activate a virtual environment (recommended)

Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

2) Install dependencies

pip install -r requirements.txt

## How To Run

Option A: Notebook (recommended)

1) Open Eval.ipynb
2) Run cells from top to bottom (the notebook will auto-build/load sampled split files under data/combined_ml_15pct/)
3) Review metrics table and visualizations
 
The notebook auto-builds/loads:
- data/combined_ml_15pct/train_sampled.csv
- data/combined_ml_15pct/test_sampled.csv
- data/combined_ml_15pct/split_stats.csv

## Evaluation Outputs

The workflow includes:
- Classification metrics (Accuracy, Precision, Recall, F1)
- False positives and false negatives comparisons
- Training and testing time comparisons
- Feature-selection overlap and importance analysis
- Optimizer convergence trends
- Prediction score distribution plots (histogram + KDE)

## Notes

- Optimization settings (agents/iterations) are configurable in Eval.ipynb.
- Higher settings generally improve search quality but increase runtime.
- Results can vary with dataset split and optimizer stochastic behavior.

## Reproducibility Tips

- Keep random_state fixed where possible
- Record package versions from requirements.txt
- Run all experiments on the same dataset split when comparing optimizers

## Troubleshooting

### CSV Loads But 'label' Column Is Missing

If you see an error like:
- `Error: Target column 'label' not found. Available columns: ['version https://git-lfs.github.com/spec/v1']...`
- `ValueError: not enough values to unpack`

your dataset files are likely Git LFS pointer files instead of the real CSV content.

Fix:

```powershell
git lfs pull origin main
```

Optional check (first lines should show CSV headers, not LFS pointer text):

```powershell
Get-Content -Path "data/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" -TotalCount 3
```

## Author

AIM coursework project
