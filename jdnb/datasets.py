import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def make_synthetic_data():
    """Generates the two 2D synthetic datasets from the paper."""
    
    # 1. Axis-aligned Data
    X1, y1 = make_classification(
        n_samples=600, n_features=2, n_informative=2,
        n_redundant=0, class_sep=1.0, random_state=42
    )

    # 2. Correlated Gaussian Data (from your PDF/code)
    np.random.seed(42)
    mean0, mean1 = np.array([-1, 0]), np.array([1, 0])
    cov0 = np.array([[2.0, 1.6], [1.6, 2.0]])
    cov1 = np.array([[2.0, -1.6], [-1.6, 2.0]])
    X0 = np.random.multivariate_normal(mean0, cov0, 300)
    X1b = np.random.multivariate_normal(mean1, cov1, 300)
    X2 = np.vstack([X0, X1b])
    y2 = np.array([0]*300 + [1]*300)

    datasets = {
        "Axis-aligned Data": (X1, y1),
        "Correlated Data": (X2, y2)
    }
    return datasets

def load_wine():
    """Loads the WineQuality dataset (binarized)."""
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df = pd.read_csv(url, sep=";")
    except Exception as e:
        print(f"Error loading Wine dataset: {e}")
        print("Please check your internet connection or the UCI URL.")
        return "WineQuality", None, None
        
    X = df.drop("quality", axis=1).values
    y = (df["quality"] >= 6).astype(int)  # binarized
    return "WineQuality", X, y

def load_banknote():
    """Loads the Banknote Authentication dataset."""
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
        df = pd.read_csv(url, header=None)
    except Exception as e:
        print(f"Error loading Banknote dataset: {e}")
        print("Please check your internet connection or the UCI URL.")
        return "Banknote", None, None
        
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return "Banknote", X, y

def find_most_correlated_pair(X):
    """Finds the two most correlated features in X."""
    df = pd.DataFrame(X)
    corr_matrix = df.corr().abs()
    
    # Mask the diagonal (self-correlation)
    np.fill_diagonal(corr_matrix.values, 0)
    
    # Find the indices of the max value
    max_corr = corr_matrix.max().max()
    max_indices = corr_matrix.where(corr_matrix == max_corr).stack().index[0]
    
    f1_idx, f2_idx = int(max_indices[0]), int(max_indices[1])
    return f1_idx, f2_idx