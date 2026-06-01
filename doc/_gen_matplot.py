"""Generate PNG figures from quaternion_matplot/quaternion_plot.py tests."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "quaternion_matplot"))

OUT = os.path.join(ROOT, "doc")
os.makedirs(OUT, exist_ok=True)

from quaternion_plot import TestQuaternionPlot  # noqa: E402

tst = TestQuaternionPlot()

cases = [
    ("matplot_single",        tst.test_quaternion_plot),
    ("matplot_multi",         tst.test_quaternion_multi_plot),
    ("matplot_norm",          tst.test_quaternion_norm),
    ("matplot_multiplication", tst.test_show_multiplication),
]

_orig_show = plt.show
_current = {"name": None}

def _save(*a, **kw):
    for num in plt.get_fignums():
        fig = plt.figure(num)
        path = os.path.join(OUT, f"{_current['name']}.png")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        print("saved", path)
    plt.close("all")

plt.show = _save

for name, fn in cases:
    _current["name"] = name
    fn()
