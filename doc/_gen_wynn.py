"""Generate one PNG frame from quaternion_wynn animation."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from quaternion_wynn.quaternion import Quaternion  # noqa: E402

OUT = os.path.join(ROOT, "doc")
os.makedirs(OUT, exist_ok=True)

# A simple still: rotate the basis triad by a sequence of quaternions
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim((-2, 2)); ax.set_ylim((-2, 2)); ax.set_zlim((-2, 2))
ax.set_title("quaternion_wynn: rotated basis (SLERP samples)")

q1 = Quaternion(axis=[0, 0, 1], degrees=0)
q2 = Quaternion(axis=[1, 1, 0], degrees=120)

basis_colors = ["r", "g", "b"]
endpoints = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]])

steps = list(Quaternion.intermediates(q1, q2, 6, include_endpoints=True))
for k, q in enumerate(steps):
    alpha = 0.25 + 0.75 * k / max(1, len(steps) - 1)
    for v, c in zip(endpoints, basis_colors):
        e = q.rotate(v)
        ax.plot([0, e[0]], [0, e[1]], [0, e[2]], c=c, alpha=alpha, lw=2)

ax.view_init(25, -60)
out = os.path.join(OUT, "wynn_slerp_basis.png")
fig.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)
plt.close(fig)
