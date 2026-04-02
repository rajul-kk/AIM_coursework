#!/usr/bin/env python
# coding: utf-8

# # Metaheuristic Performance Analysis for Intrusion Detection System
# 
# This notebook provides the complete end-to-end pipeline for evaluating the performance of a Random Forest IDS model against three metaheuristic algorithms (GA, PSO, GWO) applied for simultaneous Feature Selection and Hyperparameter Tuning.

# ## 1. Setup and Imports
# Importing required standard libraries and the custom modules created for this coursework.

# In[ ]:


import sys
import os
sys.path.append(os.path.abspath('./'))
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import sys
import os
sys.path.append(os.path.abspath('./'))
# Ensure the custom modules from the higher directory can be imported
sys.path.append(os.path.abspath(os.path.join('.')))

from preprocessing.clean import load_and_preprocess_data
from core.baseline import run_baseline
from optimizers.pso import run_pso
from optimizers.ga import run_ga
from optimizers.gwo import run_gwo

import warnings
warnings.filterwarnings('ignore')

print("Libraries successfully imported.")


# ## 2. Data Loading and Preprocessing
# Extracting the raw CICIDS2017 sub-dataset, cleaning it, and establishing a standard binary classification (Normal: 0, Attack: 1) Train/Test split.

# In[ ]:


DATA_PATH = "data/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# load_and_preprocess_data handles loading, cleaning, target binarization, stratification, and scaling.
X_train_full, X_test, y_train_full, y_test, scaler, _ = load_and_preprocess_data(DATA_PATH)

# Note: Metaheuristic optimization requires evaluating the model over and over. 
# To make the optimization finish in a reasonable time, we use a smaller, stratified subset 
# of the training data specifically for the "optimization" phase (the Fitness evaluations).
# The final models will still be evaluated against the full test set.

TRAIN_SUBSET_SIZE = 5000
if X_train_full is not None and X_train_full.shape[0] > TRAIN_SUBSET_SIZE:
    # Select a random subset to speed up training during the fitness function loops
    idx = np.random.choice(X_train_full.shape[0], TRAIN_SUBSET_SIZE, replace=False)
    X_train_subset = X_train_full[idx]
    y_train_subset = y_train_full.iloc[idx].values
else:
    X_train_subset = X_train_full
    if y_train_full is not None:
        y_train_subset = y_train_full.values

print(f"\nFull Test Data available: {X_test.shape[0]} arrays")
print(f"Optimization Training Subset: {X_train_subset.shape[0]} arrays")


# ## 3. Train Baseline Random Forest
# We establish the baseline by training a standard Random Forest classification model using all available features and the default hyperparameters.

# In[ ]:


print("Starting baseline training...")
baseline_metrics, rf_baseline = run_baseline(X_train_subset, X_test, y_train_subset, y_test)


# ## 4. Run Metaheuristic Optimizations (GA, PSO, GWO)
# We now run the three nature-inspired algorithms. For each algorithm, it will iterate through multiple generations/steps to minimize the fitness/cost function: `-(0.9 * Accuracy + 0.1 * Feature_Reduction)`.
# 
# *Note: The number of agents and iterations are currently set very low for quick testing. For your final report, increase these (e.g., 20 agents, 30 iterations).*.

# In[ ]:


NUM_AGENTS = 5
NUM_ITERATIONS = 5

print("\n===================================")
print("=== Running Genetic Algorithm ===")
print("===================================")
ga_metrics, rf_ga, ga_mask, ga_history = run_ga(X_train_subset, y_train_subset, X_test, y_test, 
                           pop_size=NUM_AGENTS, num_iterations=NUM_ITERATIONS)

print("\n=========================================")
print("=== Running Particle Swarm Optimization ===")
print("=========================================")
pso_metrics, rf_pso, pso_mask, pso_history = run_pso(X_train_subset, y_train_subset, X_test, y_test, 
                              num_particles=NUM_AGENTS, num_iterations=NUM_ITERATIONS)

print("\n===================================")
print("=== Running Grey Wolf Optimizer ===")
print("===================================")
gwo_metrics, rf_gwo, gwo_mask, gwo_history = run_gwo(X_train_subset, y_train_subset, X_test, y_test, 
                              num_wolves=NUM_AGENTS, num_iterations=NUM_ITERATIONS)


# ## 5. View Comparison Metrics
# Combining the resulting metrics dictionaries (which are uniform across all algorithms) into a Pandas DataFrame for an easy tabular comparison.

# In[ ]:


results_df = pd.DataFrame([
    {'Model': 'Baseline RF', **baseline_metrics},
    {'Model': 'GA + RF', **ga_metrics},
    {'Model': 'PSO + RF', **pso_metrics},
    {'Model': 'GWO + RF', **gwo_metrics}
])

cols = ['Model', 'accuracy', 'precision', 'recall', 'f1', 'false_positives', 'feature_count', 'train_time', 'test_time']
results_df = results_df[cols]

display(results_df)


# ## 6. Confusion Matrices Visualizations
# Visually inspecting the confusion matrices for each model gives an immediate sense of how False Positives and False Negatives are distributed.

# In[ ]:


def plot_model_cm(ax, model, X_test_filtered, y_test, title):
    y_pred = model.predict(X_test_filtered)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Baseline
plot_model_cm(axes[0, 0], rf_baseline, X_test, y_test, "Baseline RF")

# Important Note! Our optimal models were trained on feature SUBSETS.
# When sklearn trains a model, it expects the EXACT SAME number of features to be 
# passed into its `predict` function as were passed into its `fit` function.
# Because `rf_ga.n_features_in_` is the optimal subset length, we just slice the beginning
# features for visualization (or if we saved our boolean masks directly, we would apply them here).
# Since our wrappers don't currently expose the final boolean mask, we simply slice X_test to match n_features_in_.

# 2. Genetic Algorithm (GA)
plot_model_cm(axes[0, 1], rf_ga, X_test[:, ga_mask], y_test, "GA + RF")

# 3. Particle Swarm Optimization (PSO)
plot_model_cm(axes[1, 0], rf_pso, X_test[:, pso_mask], y_test, "PSO + RF")

# 4. Grey Wolf Optimizer (GWO)
plot_model_cm(axes[1, 1], rf_gwo, X_test[:, gwo_mask], y_test, "GWO + RF")

plt.tight_layout()
plt.show()

