import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, log_loss, f1_score, roc_auc_score,
    precision_score, recall_score
)

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jdnb.model import JDNB
from jdnb.utils import class_stats, offdiag_energy, kl_divergence_cov
from jdnb.datasets import load_wine, load_banknote

def run_experiment(name, X, y):
    """Helper to run GNB vs JDNB for a single dataset."""
    if X is None:
        print(f"Skipping {name}, data not loaded.")
        return None
        
    print(f"\n--- Running Experiment on {name} ({X.shape[1]} features) ---")
    
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    
    # Get Base Energy
    classes, mus, Sigmas, priors = class_stats(Xtr, ytr)
    base_energy = offdiag_energy(Sigmas)

    # GaussianNB
    gnb = GaussianNB().fit(Xtr, ytr)
    ypred_gnb = gnb.predict(Xte)
    p_gnb = gnb.predict_proba(Xte)

    # JD-Naive Bayes
    jdnb = JDNB(lr=5e-3, iters=500).fit(Xtr, ytr)
    ypred_jd = jdnb.predict(Xte)
    p_jd = jdnb.predict_proba(Xte)

    # Get Metrics
    metrics = [
        name,
        accuracy_score(yte, ypred_gnb), accuracy_score(yte, ypred_jd),
        f1_score(yte, ypred_gnb), f1_score(yte, ypred_jd),
        precision_score(yte, ypred_gnb), precision_score(yte, ypred_jd),
        recall_score(yte, ypred_gnb), recall_score(yte, ypred_jd),
        roc_auc_score(yte, p_gnb[:, 1]), roc_auc_score(yte, p_jd[:, 1]),
        log_loss(yte, p_gnb), log_loss(yte, p_jd)
    ]

    # Post-JDNB Energy
    _, _, Sigmas_z, _ = class_stats(Xtr @ jdnb.W, ytr)
    jd_energy = offdiag_energy(Sigmas_z)
    kl_val = kl_divergence_cov(Sigmas[0], Sigmas_z[0])
    metrics.extend([base_energy, jd_energy, kl_val])
    
    return metrics

def main():
    print("="*50)
    print(" RUN 2: REAL HIGH-DIMENSIONAL DATASETS")
    print("="*50 + "\n")
    
    sns.set(style="whitegrid", context="notebook")
    
    datasets_to_load = [load_wine, load_banknote]
    results = []
    
    for loader in datasets_to_load:
        name, X, y = loader()
        metrics = run_experiment(name, X, y)
        if metrics:
            results.append(metrics)

    # Summary Metrics Table
    columns = [
        "Dataset", "Acc_GNB", "Acc_JDNB", "F1_GNB", "F1_JDNB",
        "Prec_GNB", "Prec_JDNB", "Rec_GNB", "Rec_JDNB", "AUC_GNB", "AUC_JDNB",
        "LogLoss_GNB", "LogLoss_JDNB", "OffDiag_Energy_Before", "OffDiag_Energy_After", "KL_Divergence"
    ]
    df = pd.DataFrame(results, columns=columns)
    print("\n=== RUN 2: EXTENDED PERFORMANCE SUMMARY ===\n")
    print(df.round(4).to_string(index=False))

    # Visualization of Off-Diagonal Energy
    fig_energy = plt.figure(figsize=(7, 5))
    bar_data = df.melt(id_vars="Dataset", value_vars=["OffDiag_Energy_Before", "OffDiag_Energy_After"],
                      var_name="Stage", value_name="OffDiag_Energy")
    sns.barplot(x="Dataset", y="OffDiag_Energy", hue="Stage", data=bar_data, palette="muted")
    plt.yscale("log")
    plt.ylabel("Off-Diagonal Energy (log scale)")
    plt.title("RUN 2: Reduction in Off-Diagonal Covariance Energy (Log Scale)")
    plt.tight_layout()
    
    # Save Energy Plot
    outputs_dir = os.path.join(project_root, 'outputs')
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    energy_plot_filename = os.path.join(outputs_dir, 'run2_real_world_energy.png')
    fig_energy.savefig(energy_plot_filename)
    print(f"\nEnergy plot saved to {energy_plot_filename}")
    plt.show()

if __name__ == "__main__":
    main()
