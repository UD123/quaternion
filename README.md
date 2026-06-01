# Quaternion

A study repository with **three independent Python implementations of quaternion
algebra**, each taking a different approach and offering its own visualization
of rotations in 3D space.

| Folder | Core idea | Visualization |
| --- | --- | --- |
| [`quaternion_wynn/`](quaternion_wynn/) | Full-featured port of [`pyquaternion`](https://github.com/KieranWynn/pyquaternion) by Kieran Wynn | `matplotlib` animation of a rotating triad |
| [`quaternion_matplot/`](quaternion_matplot/) | Subclass of the Wynn `Quaternion` that adds a "flag"-style 3D plot | Static `matplotlib` 3D plot (arrow + flag) |
| [`quaternion_simplega/`](quaternion_simplega/) | Minimal from-scratch implementation (port of `SimpleGA.jl/quaternions.jl`) with rich didactic plotting | 7-figure didactic gallery: conjugate, rotation, SLERP, composition, flags, sandwich product |

All three packages are self-contained Python modules; pick one depending on
whether you want a production-style API (`quaternion_wynn`), a simple plotting
helper (`quaternion_matplot`), or detailed teaching figures
(`quaternion_simplega`).

---

## Flag encoding of a quaternion

Both visualizing packages draw a unit (or near-unit) quaternion

$$q = w + x\,\mathbf{i} + y\,\mathbf{j} + z\,\mathbf{k}, \qquad
  \|q\|^2 = w^2 + x^2 + y^2 + z^2$$

as a small **flag** in 3D. A quaternion has four real numbers, so the flag
needs four visual degrees of freedom. They are mapped as follows:

| Quaternion quantity | Geometric meaning | Flag element |
| --- | --- | --- |
| Imaginary part `(x, y, z)` | Rotation axis $\hat{n} = (x,y,z)/\sin(\theta/2)$ — 2 d.o.f. (direction on the unit sphere) | **Pole**: a line from the origin pointing along $\hat{n}$ |
| Scalar part `w` | Rotation angle $\theta = 2\arccos(w)$ — 1 d.o.f. | **Pennant** rotated by $\theta$ around the pole (and the labelled arc next to the pole tip) |
| Norm $\|q\|$ | Magnitude — 1 d.o.f. (= 1 for a rotation) | **Pole length** (only in `simplega`'s `add_unnormalized_flag` and in `matplot`'s `QuaternionPlot.plot`); unit quaternions use a fixed length |
| Sign of $w$ | The double-cover ambiguity ($q$ and $-q$ are the same rotation) | Both packages normalize to $w \ge 0$, so the pennant always sweeps the *short* way around the pole |

So in one picture you can read off: **where the rotation axis points** (the
pole), **how much the rotation is** (the pennant's angular offset around the
pole, also written next to the small arc), and **how long the quaternion is**
(the pole length, when not normalized).

### `simplega`'s `add_flag` — the explicit construction

Given $\hat{n}$ and $\theta$:

1. Pick any unit vector $u \perp \hat{n}$, and let $v = \hat{n} \times u$.
   Together $(u, v, \hat{n})$ form an orthonormal frame; $u$ marks the
   $\theta = 0$ reference.
2. The pennant tip is placed at the pole tip $P = \hat{n}$ and points in the
   direction $\cos\theta \cdot u + \sin\theta \cdot v$, i.e. it has rotated by
   $\theta$ around the pole.
3. A dashed gray line shows $u$ (the $\theta = 0$ reference) and a small
   coloured arc sweeps from $u$ to the pennant, labelled with $\theta$ in
   degrees.

### `matplot`'s `QuaternionPlot.plot` — the implicit construction

`matplot` does not build the pennant from $(\hat{n}, \theta)$ directly. It
draws a flat reference flag in the $x$–$z$ plane (pole along $+x$, triangular
pennant at the pole tip) and then **rotates the whole flag by $q$** using
`q.rotate(point)`. The same four d.o.f. are still visible — pole direction =
axis, pennant orientation around the pole = angle, pole length =
`self.norm` — but the angle is shown *implicitly* by where the pennant ends
up rather than via an explicit arc.

### Reading examples

* **Figure 1 (`doc/simplega_fig1.png`)** — two flags for $q = $ 90° about
  $\hat{z}$ and its conjugate $q^\dagger$. Same pole (the $z$-axis), opposite
  pennant directions, arcs labelled `+90°` and `−90°`.
* **Figure 5 (`doc/simplega_fig5.png`)** — three unit quaternions
  (90° $\hat{z}$, 60° $\hat{x}$, 120° $(\hat{x}+\hat{y})$). The three poles
  give you the rotation axes at a glance; the three arcs/pennants give the
  angles.
* **Figure 6 (`doc/simplega_fig6.png`)** — four *unnormalized* quaternions.
  Pole lengths visibly differ ($\|q\|$ from ≈ 0.36 up to ≈ 1.72) while the
  pennant still encodes $\theta = 2\arccos(w/\|q\|)$.
* **`doc/matplot_multiplication.png`** — same flag style, used to read off
  multiplication geometrically: $q_3 = q_2 \cdot q_1$, $q_4 = q_2^{-1}$,
  $q_5 = q_3 \cdot q_4 = q_2 \cdot q_1 \cdot q_2^{-1}$ — the conjugation that
  rotates $q_1$'s axis by $q_2$.

### Minimal code

```python
# simplega: explicit flag, normalized
import math
from quaternion_simplega.quaternions import Quaternion
from quaternion_simplega.quaternion_visualization import QuaternionVisualizer

q = Quaternion(math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4))  # 90° about z
(QuaternionVisualizer().draw_axes().add_unit_sphere().set_limits(1.5)
    .add_flag(q, color="steelblue", label="q  (90° z)")
    .add_flag(q.conj(), color="tomato", label="q†  (−90° z)")
    .legend().show())

# simplega: pole length encodes the norm
from quaternion_simplega.quaternions import Quaternion
from quaternion_simplega.quaternion_visualization import visualize_unnormalized_flag
visualize_unnormalized_flag(Quaternion(1.5, 0.6, 0.0, 0.6))   # ‖q‖ ≈ 1.72

# matplot: implicit flag (rotated reference flag)
from quaternion_matplot.quaternion_plot import QuaternionPlot
import matplotlib.pyplot as plt
ax = QuaternionPlot(axis=[1, 0, 0], degrees=90).plot(clr="r")
ax = QuaternionPlot(axis=[0, 1, 0], degrees=90).plot(ax, clr="g")
plt.show()
```

---

## `quaternion_wynn/` — pyquaternion port

A near-verbatim copy of Kieran Wynn's `pyquaternion` module. Supports
construction from axis/angle, rotation matrices, arrays, SLERP, exp/log,
random unit quaternions, and rotation of 3-vectors.

### Usage

```python
from quaternion_wynn.quaternion import Quaternion

# +90° about the +y axis
q = Quaternion(axis=[0, 1, 0], degrees=90)

v = [0, 0, 4]
print(q.rotate(v))                     # → [4, 0, 0]

# SLERP between two orientations
q0 = Quaternion(axis=[0, 1, 0], angle=0)
for qi in Quaternion.intermediates(q0, q, 9, include_endpoints=True):
    print(round(qi.degrees, 2), qi.axis)
```

Run the interactive animation (rotating coordinate triad):

```powershell
python quaternion_wynn/quaternion_animation.py
```

A still snapshot of several SLERP samples applied to the basis triad:

![wynn slerp basis](doc/wynn_slerp_basis.png)

The folder also contains `algebra.py` / `r020.py`, an alternative
Clifford-algebra `Cl(0,2,0)` view of the same algebra.

---

## `quaternion_matplot/` — Wynn quaternion + flag plot

Reuses the same `Quaternion` class as `quaternion_wynn`, and adds a
`QuaternionPlot` subclass that draws each quaternion as an oriented **flag** in
3D: a pole along the rotation axis with a small pennant whose orientation
encodes the rotation angle.

### Usage

```python
from quaternion_matplot.quaternion_plot import QuaternionPlot
import matplotlib.pyplot as plt

q1 = QuaternionPlot(axis=[1, 0, 0], degrees=90)
q2 = QuaternionPlot(axis=[0, 1, 0], degrees=90)
q3 = q2 * q1            # composition

ax = q1.plot(clr="r")
ax = q2.plot(ax, clr="g")
ax = q3.plot(ax, clr="b")
plt.show()
```

Run the bundled examples:

```powershell
python quaternion_matplot/quaternion_plot.py    # built-in test cases
python quaternion_matplot/demo1.py              # text demo (rotation + SLERP)
python quaternion_matplot/demo2.py              # matplotlib animation
```

### Figures

| Single quaternion | Several axes/angles |
| --- | --- |
| ![single](doc/matplot_single.png) | ![multi](doc/matplot_multi.png) |

| Non-unit axes (norm effect) | Multiplication: `q1`, `q2`, `q2·q1`, `q2⁻¹`, `q2·q1·q2⁻¹` |
| --- | --- |
| ![norm](doc/matplot_norm.png) | ![mul](doc/matplot_multiplication.png) |

---

## `quaternion_simplega/` — minimal quaternion + didactic gallery

A small, dependency-light `Quaternion` class (`__slots__`-based) ported from
`SimpleGA.jl/src/quaternions.jl`, paired with the most extensive visualization
in this repo (`quaternion_visualization.py`).

The visualizer provides:

- `axis_angle(q)`, `rotate_vec(q, v)`, `slerp(q1, q2, t)` geometry helpers
- a chainable `QuaternionVisualizer` class
  (`add_quaternion`, `add_flag`, `add_slerp`, `add_composition`,
  `add_sandwich_rotation`, `add_unit_sphere`, …)
- convenience factories: `visualize_quaternion`, `visualize_flag`,
  `visualize_slerp`, `visualize_sandwich_rotation`,
  `visualize_unnormalized_flag`.

### Usage

```python
import math
from quaternion_simplega.quaternions import Quaternion
from quaternion_simplega.quaternion_visualization import QuaternionVisualizer

# Build a unit quaternion (90° about z)
s = math.sin(math.pi / 4)
q = Quaternion(math.cos(math.pi / 4), 0.0, 0.0, s)

(QuaternionVisualizer(title="q (90° z) and q†")
    .draw_axes()
    .add_unit_sphere()
    .set_limits(1.5)
    .add_flag(q,        color="steelblue", label="q")
    .add_flag(q.conj(), color="tomato",    label="q†")
    .legend()
    .show())
```

Run the full 7-figure gallery:

```powershell
python quaternion_simplega/quaternion_visualization.py
```

### Figures

| `q` and conjugate `q†` | Rotation of basis vectors |
| --- | --- |
| ![fig1](doc/simplega_fig1.png) | ![fig2](doc/simplega_fig2.png) |

| SLERP path as a sequence of flags | Composition `q₁`, `q₂`, `q₁·q₂` |
| --- | --- |
| ![fig3](doc/simplega_fig3.png) | ![fig4](doc/simplega_fig4.png) |

| Three unit quaternions as flags | Unnormalized flags (pole length = ‖q‖) |
| --- | --- |
| ![fig5](doc/simplega_fig5.png) | ![fig6](doc/simplega_fig6.png) |

Sandwich product `r = q · p · q†` with non-unit quaternions:

![fig7](doc/simplega_fig7.png)

---

## Installation (Windows / PowerShell)

```powershell
# create a virtual environment
python -m venv C:\Users\<you>\Documents\Envs\quat
C:\Users\<you>\Documents\Envs\quat\Scripts\Activate.ps1

# the three packages only need numpy + matplotlib
pip install numpy matplotlib
```

Then run any of the scripts above from the repository root.

### Regenerating the figures in `doc/`

The PNGs in [`doc/`](doc/) are produced by the small helper scripts
`doc/_gen_simplega.py`, `doc/_gen_matplot.py`, and `doc/_gen_wynn.py`:

```powershell
$env:PYTHONIOENCODING="utf-8"
python doc\_gen_simplega.py
python doc\_gen_matplot.py
python doc\_gen_wynn.py
```

---

## Credits

- **pyquaternion** — Kieran Wynn,
  <https://github.com/KieranWynn/pyquaternion> (MIT). Used by
  `quaternion_wynn/` and `quaternion_matplot/`.
- **SimpleGA.jl** — basis for the minimal quaternion class in
  `quaternion_simplega/`.
- **bivector.net** — Clifford-algebra `Cl(0,2,0)` generator used by
  `quaternion_wynn/algebra.py`.