import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

from jdnb.datasets import make_synthetic_data
from jdnb.utils import class_stats, jd_loss_and_grad, orthonormalize

sns.set(style="whitegrid")

# Load synthetic correlated dataset
datasets = make_synthetic_data()
X, y = datasets["Correlated Data"]

# Gradient descent snapshots
W = np.eye(2)
Ws = [W.copy()]

for _ in range(200):
    classes, mus, Sigmas, priors = class_stats(X, y)
    loss, G = jd_loss_and_grad(W, Sigmas)
    W -= 5e-3 * G
    W = orthonormalize(W)
    Ws.append(W.copy())

# Animation
fig, ax = plt.subplots(figsize=(6, 6))
scatter = ax.scatter([], [], s=20)
title = ax.set_title("", fontsize=12)
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)

def update(frame):
    W = Ws[frame]
    Z = X @ W
    scatter.set_offsets(Z)
    scatter.set_array(y)
    title.set_text(f"JD-NB Transformation — Iter {frame}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    return scatter, title

ani = FuncAnimation(fig, update, frames=len(Ws), interval=60)
ani.save("outputs/jdnb_mechanism.gif", dpi=120, writer="pillow")

print("Animation saved as jdnb_mechanism.gif")
