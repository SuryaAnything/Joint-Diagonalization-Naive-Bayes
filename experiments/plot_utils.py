import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_decision_boundary(model, X, y, ax, title):
    """Plots the decision boundary for a 2D dataset."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    try:
        # Get probabilities for the grid
        probs = model.predict_proba(grid)
        
        # Use P(class_1) for the contour plot
        Z = probs[:, 1].reshape(xx.shape)
        cmap = "coolwarm"
             
        ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.4)
        
    except Exception as e:
        print(f"Could not plot contourf: {e}")

    # Plot data points
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, s=45, edgecolor="k", ax=ax, legend=False)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())