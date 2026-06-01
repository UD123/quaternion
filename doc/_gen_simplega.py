"""Generate PNG figures from quaternion_simplega/quaternion_visualization.py."""
import os, sys, runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "quaternion_simplega"))

OUT = os.path.join(ROOT, "doc")
os.makedirs(OUT, exist_ok=True)

_orig_show = plt.show
_counter = {"i": 0}

def _save_all(*a, **kw):
    for num in plt.get_fignums():
        _counter["i"] += 1
        fig = plt.figure(num)
        path = os.path.join(OUT, f"simplega_fig{_counter['i']}.png")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        print("saved", path)
    plt.close("all")

plt.show = _save_all

runpy.run_path(
    os.path.join(ROOT, "quaternion_simplega", "quaternion_visualization.py"),
    run_name="__main__",
)
