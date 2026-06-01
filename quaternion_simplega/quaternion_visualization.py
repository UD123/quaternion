"""
3D visualization of unit quaternions as rotations using matplotlib.

Quaternion geometry
-------------------
A unit quaternion  q = w + xi + yj + zk  (w²+x²+y²+z²=1) encodes a rotation:
    axis  n̂  =  (x, y, z) / sin(θ/2)
    angle θ  =  2 · arccos(w)

The sandwich product  q * p * q†  rotates a pure quaternion p (=3-D vector).
The conjugate q† = w - xi - yj - zk  is the inverse rotation (same axis, -θ).
Note: q and -q encode the *same* rotation (double cover of SO(3)).

Visualisation elements
-----------------------
• Rotation axis arrow  – direction of the imaginary part (x, y, z)
• Angle arc            – arc of radius r around the axis sweeping angle θ
• Coordinate frame     – original (gray) and rotated (coloured) i/j/k arrows
• Conjugate q†         – same axis, arc reversed; rotated frame shown in a
                         contrasting colour
• SLERP path           – smooth interpolation between two orientations applied
                         to a reference vector
• Rotation composition – q1, q2 and their product q1·q2 drawn simultaneously

Run as a script
---------------
    python simplega/quaternion_visualization.py
"""

import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Allow `python simplega/quaternion_visualization.py` from the repo root
try:
    from quaternions import Quaternion
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from quaternions import Quaternion


# ---------------------------------------------------------------------------
# Pure-geometry helpers (no matplotlib)
# ---------------------------------------------------------------------------

def _unit(q: Quaternion) -> Quaternion:
    """Return a unit quaternion; raise if q ≈ 0."""
    n = q.norm()
    if n < 1e-14:
        raise ValueError("Cannot normalise a zero quaternion.")
    return Quaternion(q.w / n, q.x / n, q.y / n, q.z / n)


def axis_angle(q: Quaternion):
    """
    Decompose a unit quaternion into (axis, angle).

    Returns
    -------
    axis  : np.ndarray, shape (3,)   unit vector
    angle : float                    rotation angle in [0, 2π)
    """
    q = _unit(q)
    # Clamp for numerical safety; always use positive-w representative
    if q.w < 0:
        q = Quaternion(-q.w, -q.x, -q.y, -q.z)
    w = min(1.0, max(-1.0, q.w))
    angle = 2.0 * math.acos(w)
    sin_half = math.sqrt(max(0.0, 1.0 - w * w))
    if sin_half < 1e-10:                     # near-identity
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = np.array([q.x, q.y, q.z]) / sin_half
    return axis / np.linalg.norm(axis), angle


def rotate_vec(q: Quaternion, v: np.ndarray) -> np.ndarray:
    """
    Rotate 3-D vector v by unit quaternion q using  q * p * q†.
    """
    q = _unit(q)
    p = Quaternion(0.0, float(v[0]), float(v[1]), float(v[2]))
    r = q * p * q.conj()
    return np.array([r.x, r.y, r.z])


def slerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
    """
    Spherical linear interpolation between unit quaternions q1 and q2.
    Always takes the shorter arc (dot product clamped ≥ 0).
    """
    q1, q2 = _unit(q1), _unit(q2)
    d = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z
    if d < 0.0:                      # take shorter path
        q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
        d = -d
    d = min(1.0, d)
    theta = math.acos(d)
    if theta < 1e-10:                # nearly identical
        w = 1.0 - t
        return Quaternion(w*q1.w + t*q2.w, w*q1.x + t*q2.x,
                          w*q1.y + t*q2.y, w*q1.z + t*q2.z)
    s = math.sin(theta)
    a = math.sin((1.0 - t) * theta) / s
    b = math.sin(t * theta) / s
    return Quaternion(a*q1.w + b*q2.w, a*q1.x + b*q2.x,
                      a*q1.y + b*q2.y, a*q1.z + b*q2.z)


def _plane_frame(normal: np.ndarray):
    """Orthonormal (u, v) spanning the plane ⊥ to normal, with u×v = n̂."""
    n = normal / (np.linalg.norm(normal) + 1e-15)
    ref = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    u = np.cross(n, ref);  u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


# ---------------------------------------------------------------------------
# Main visualiser class
# ---------------------------------------------------------------------------

class QuaternionVisualizer:
    """
    3D visualiser for unit quaternions as rotations.

    Each quaternion is rendered as:
    ● A rotation-axis arrow (direction of the imaginary part).
    ● An angle-arc sweeping θ around that axis.
    ● A coordinate frame showing where i, j, k land after the rotation.

    The conjugate q† (inverse rotation) uses the same axis with the arc
    reversed and a contrasting colour.

    Parameters
    ----------
    figsize : tuple
    title   : str
    elev, azim : float   initial 3-D viewing angles
    """

    _FRAME_COLORS = ("crimson", "forestgreen", "royalblue")  # i, j, k
    _FRAME_LABELS = ("$\\hat{x}$", "$\\hat{y}$", "$\\hat{z}$")
    _FRAME_VECS   = (np.array([1.,0.,0.]),
                     np.array([0.,1.,0.]),
                     np.array([0.,0.,1.]))

    def __init__(self, figsize=(11, 9),
                 title: str = "Quaternion Visualisation",
                 elev: float = 22.0, azim: float = -55.0):
        self.fig = plt.figure(figsize=figsize)
        self.ax  = self.fig.add_subplot(111, projection="3d")
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.set_xlabel("$x$", fontsize=12, labelpad=8)
        self.ax.set_ylabel("$y$", fontsize=12, labelpad=8)
        self.ax.set_zlabel("$z$", fontsize=12, labelpad=8)
        self.ax.set_title(title, fontsize=13, pad=12)

    # ------------------------------------------------------------------
    # Low-level drawing helpers
    # ------------------------------------------------------------------

    def _arrow(self, origin, direction, color, label=None,
               lw=2.2, ratio=0.15, length=None):
        """Draw a single arrow."""
        d = np.asarray(direction, dtype=float)
        if length is not None:
            n = np.linalg.norm(d)
            if n > 1e-14:
                d = d / n * length
        kw = dict(color=color, linewidth=lw, arrow_length_ratio=ratio)
        if label:
            kw["label"] = label
        o = np.asarray(origin, dtype=float)
        self.ax.quiver(o[0], o[1], o[2], d[0], d[1], d[2], **kw)

    def _angle_arc(self, axis: np.ndarray, angle: float,
                   color, sign: float = 1.0,
                   radius: float = 0.45, n_pts: int = 60):
        """
        Draw an arc of `|angle|` radians around `axis`, plus an arrowhead.
        sign=+1  →  CCW from the axis (positive rotation)
        sign=-1  →  CW  (conjugate / inverse rotation)
        """
        if abs(angle) < 1e-6:
            return
        u, v = _plane_frame(axis)
        t = np.linspace(0.0, sign * angle, n_pts)
        pts = (np.outer(np.cos(t), radius * u) +
               np.outer(np.sin(t), radius * v))
        self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                     color=color, lw=2.4, alpha=0.9)
        # Arrowhead at the tip
        tang = pts[-1] - pts[-3]
        tang_n = np.linalg.norm(tang)
        if tang_n > 1e-14:
            tang /= tang_n
            h = 0.06 * radius
            self.ax.quiver(*pts[-2], *(tang * h),
                           color=color, lw=2.2, arrow_length_ratio=0.99)

    def _angle_sector(self, axis: np.ndarray, angle: float,
                      color, sign: float = 1.0,
                      radius: float = 0.45, alpha: float = 0.18,
                      n_pts: int = 48):
        """Fill the pie-slice sector showing the rotation angle."""
        if abs(angle) < 1e-6:
            return
        u, v = _plane_frame(axis)
        t = np.linspace(0.0, sign * angle, n_pts)
        rim = (np.outer(np.cos(t), radius * u) +
               np.outer(np.sin(t), radius * v))
        # sector = origin + rim points (closed polygon)
        origin = np.zeros((1, 3))
        verts = np.vstack([origin, rim, origin])
        poly = Poly3DCollection([verts], alpha=alpha,
                                facecolor=color, edgecolor="none")
        self.ax.add_collection3d(poly)

    def _frame(self, q: Quaternion, color,
               alpha: float = 0.9, length: float = 0.65,
               draw_original: bool = False):
        """
        Draw the rotated coordinate frame.
        Optionally also draw the un-rotated frame in gray.
        """
        if draw_original:
            for vec, lbl, col in zip(self._FRAME_VECS,
                                     self._FRAME_LABELS,
                                     self._FRAME_COLORS):
                self._arrow((0,0,0), vec, color="lightgray",
                             lw=1.2, ratio=0.12, length=length * 0.7)

        for vec, lbl, col in zip(self._FRAME_VECS,
                                  self._FRAME_LABELS,
                                  self._FRAME_COLORS):
            rotated = rotate_vec(q, vec)
            self._arrow((0,0,0), rotated, color=color,
                        lw=1.8, ratio=0.13, length=length)
            # label at the tip
            tip = rotated / (np.linalg.norm(rotated) + 1e-15) * (length + 0.1)
            self.ax.text(*tip, lbl, color=color, fontsize=9, ha="center")

    # ------------------------------------------------------------------
    # Public add_* methods
    # ------------------------------------------------------------------

    def add_quaternion(self, q: Quaternion, *,
                       color: str = "steelblue",
                       label: str | None = None,
                       show_conjugate: bool = False,
                       conjugate_color: str = "tomato",
                       conjugate_label: str | None = None,
                       show_frame: bool = True,
                       show_original_frame: bool = True,
                       show_sector: bool = True,
                       axis_length: float = 1.1,
                       arc_radius: float = 0.45):
        """
        Visualise a unit quaternion as an oriented rotation in 3-D.

        Draws:
        - Rotation axis arrow (imaginary-part direction)
        - Angle arc (sector) showing θ = 2·arccos(w)
        - Rotated coordinate frame R(x̂), R(ŷ), R(ẑ)
        - Optionally the same for the conjugate q†

        Parameters
        ----------
        q                  : unit quaternion
        color              : colour for q
        label              : legend label for the axis arrow of q
        show_conjugate     : also draw q† (inverse rotation)
        conjugate_color    : colour for q†
        conjugate_label    : legend label for q†'s axis arrow
        show_frame         : draw the rotated coordinate frame
        show_original_frame: draw original (un-rotated) frame in gray
        show_sector        : fill the angle pie-slice
        axis_length        : length of the axis arrow
        arc_radius         : radius of the angle arc / sector

        Returns
        -------
        self  (for chaining)
        """
        q = _unit(q)
        ax_vec, angle = axis_angle(q)

        # ── rotation axis ──────────────────────────────────────────────
        self._arrow((0,0,0), ax_vec, color, label=label,
                    lw=2.6, length=axis_length)

        # ── angle sector + arc ─────────────────────────────────────────
        if show_sector:
            self._angle_sector(ax_vec, angle, color, sign=+1,
                               radius=arc_radius, alpha=0.22)
        self._angle_arc(ax_vec, angle, color, sign=+1, radius=arc_radius)

        # ── angle annotation ───────────────────────────────────────────
        if angle > 1e-4:
            u, _ = _plane_frame(ax_vec)
            mid_t = angle / 2
            label_pos = arc_radius * 1.25 * (math.cos(mid_t) * u +
                                              math.sin(mid_t) * _plane_frame(ax_vec)[1])
            deg = math.degrees(angle)
            self.ax.text(*label_pos, f"{deg:.1f}°",
                         color=color, fontsize=9, ha="center")

        # ── rotated coordinate frame ────────────────────────────────────
        if show_frame:
            self._frame(q, color,
                        draw_original=show_original_frame)

        # ── conjugate q† ───────────────────────────────────────────────
        if show_conjugate:
            qc = q.conj()
            clbl = conjugate_label or (f"{label}†" if label else "q†")
            # axis is the same; arc goes the other way (sign = -1)
            self._arrow((0,0,0), ax_vec, conjugate_color, label=clbl,
                        lw=2.2, length=axis_length * 0.92)
            if show_sector:
                self._angle_sector(ax_vec, angle, conjugate_color, sign=-1,
                                   radius=arc_radius * 0.88, alpha=0.22)
            self._angle_arc(ax_vec, angle, conjugate_color, sign=-1,
                            radius=arc_radius * 0.88)
            if show_frame:
                self._frame(qc, conjugate_color,
                            draw_original=False)

        return self

    def add_rotation_effect(self, q: Quaternion,
                             vectors: list,
                             *,
                             before_color: str = "lightgray",
                             after_color: str = "steelblue",
                             labels: list | None = None,
                             show_arc: bool = True,
                             arc_steps: int = 30):
        """
        Show a set of 3-D vectors before and after rotation by q.

        For each vector v:
        - Draws v in `before_color`
        - Draws R(v) in `after_color`
        - Draws a dashed arc connecting v → R(v)

        Parameters
        ----------
        q        : unit quaternion
        vectors  : list of array-like of shape (3,)
        labels   : optional list of legend labels (same length as vectors)
        show_arc : draw the curved path from v to R(v)
        """
        q = _unit(q)
        ax_vec, angle = axis_angle(q)
        labels = labels or [None] * len(vectors)

        for v_raw, lbl in zip(vectors, labels):
            v = np.asarray(v_raw, dtype=float)
            rv = rotate_vec(q, v)

            # before
            self._arrow((0,0,0), v, before_color, lw=1.6, ratio=0.12)
            # after
            self._arrow((0,0,0), rv, after_color, label=lbl,
                        lw=2.2, ratio=0.14)

            if show_arc and angle > 1e-4:
                # Sweep v through the rotation incrementally
                pts = np.array([
                    rotate_vec(
                        slerp(Quaternion.one(), q, k / arc_steps), v
                    )
                    for k in range(arc_steps + 1)
                ])
                self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                             color=after_color, lw=1.5,
                             linestyle="--", alpha=0.7)
        return self

    def add_slerp(self, q1: Quaternion, q2: Quaternion,
                  reference_vec: np.ndarray | None = None,
                  *,
                  n_steps: int = 16,
                  color_start: str = "steelblue",
                  color_end: str = "tomato",
                  label_start: str | None = None,
                  label_end: str | None = None):
        """
        Visualise the SLERP path from q1 to q2.

        Draws the trajectory swept by `reference_vec` (default: x̂) as the
        orientation interpolates smoothly from q1 to q2.  The endpoints are
        marked with solid arrows; intermediate steps fade between the two colours.

        Parameters
        ----------
        reference_vec : (3,) array  – vector to track along the path
        """
        if reference_vec is None:
            reference_vec = np.array([1.0, 0.0, 0.0])
        v = np.asarray(reference_vec, dtype=float)
        v /= np.linalg.norm(v) + 1e-15

        sc = np.array(plt.matplotlib.colors.to_rgb(color_start))
        ec = np.array(plt.matplotlib.colors.to_rgb(color_end))

        pts = []
        for k in range(n_steps + 1):
            t = k / n_steps
            qi = slerp(q1, q2, t)
            pts.append(rotate_vec(qi, v))
        pts = np.array(pts)

        # Path
        for k in range(n_steps):
            col = tuple(sc + (k / n_steps) * (ec - sc))
            self.ax.plot(pts[k:k+2, 0], pts[k:k+2, 1], pts[k:k+2, 2],
                         color=col, lw=2.5, alpha=0.85)

        # Endpoint arrows
        self._arrow((0,0,0), pts[0],  color_start,
                    label=label_start, lw=2.5, ratio=0.15)
        self._arrow((0,0,0), pts[-1], color_end,
                    label=label_end,   lw=2.5, ratio=0.15)
        return self

    def add_composition(self, q1: Quaternion, q2: Quaternion,
                        *,
                        color1: str = "steelblue",
                        color2: str = "seagreen",
                        color_prod: str = "darkorange",
                        label1: str = "q₁",
                        label2: str = "q₂",
                        label_prod: str = "q₁·q₂",
                        arc_radius: float = 0.45):
        """
        Draw q1, q2, and their product q1·q2 simultaneously.

        Useful for understanding how rotations compose.
        """
        q1, q2 = _unit(q1), _unit(q2)
        qp = _unit(q1 * q2)

        for q, col, lbl in [(q1, color1, label1),
                             (q2, color2, label2),
                             (qp, color_prod, label_prod)]:
            ax_v, ang = axis_angle(q)
            self._arrow((0,0,0), ax_v, col, label=lbl,
                        lw=2.4, length=1.05)
            self._angle_arc(ax_v, ang, col, sign=+1,
                            radius=arc_radius)
            self._angle_sector(ax_v, ang, col, sign=+1,
                               radius=arc_radius, alpha=0.18)
        return self

    def add_flag(self, q: Quaternion, *,
                 color: str = "steelblue",
                 label: str | None = None,
                 pole_length: float = 1.0,
                 flag_len: float = 0.2,
                 flag_width: float = 0.40,
                 alpha: float = 0.72,
                 show_arc: bool = True,
                 show_reference: bool = True,
                 arc_radius: float = 0.28):
        """
        Draw a unit quaternion as a flagpole + triangular pennant.

        Encoding
        --------
        • **Pole**   : arrow along the rotation axis  n̂ = (x,y,z)/sin(θ/2).
        • **Pennant**: triangle attached at the pole tip.  Its tip points in
                       the direction  cos(θ)·u + sin(θ)·v  in the plane ⊥ to
                       the axis, where u is a fixed reference direction (θ=0).
                       The pennant angle around the axis encodes θ = 2·arccos(w).
        • **Arc**    : small arc at the pole tip sweeping from the reference
                       direction u to the pennant tip, labelled with θ in degrees.
        • **Ref line**: dashed gray line showing the θ=0 reference direction.

        Parameters
        ----------
        q              : unit quaternion
        color          : colour for pole, pennant and arc
        label          : legend label (attached to the pole arrow)
        pole_length    : length of the pole arrow
        flag_len       : distance from pole tip to pennant tip
        flag_width     : base width of the triangular pennant
        alpha          : opacity of the pennant face
        show_arc       : draw the angle arc at the pole tip
        show_reference : draw a dashed reference line at θ=0
        arc_radius     : radius of the angle arc

        Returns
        -------
        self  (for chaining)
        """
        q = _unit(q)
        ax_vec, angle = axis_angle(q)

        # Orthonormal frame ⊥ to the axis; u is the θ=0 reference direction
        u, v = _plane_frame(ax_vec)

        # Pennant direction at angle θ from reference u
        cos_t, sin_t = math.cos(angle), math.sin(angle)
        fd      =  cos_t * u + sin_t * v          # toward pennant tip
        fd_perp =  cos_t * v - sin_t * u          # ⊥ to fd in the same plane

        # Pole tip position
        P = ax_vec * pole_length

        # ── pole (plain line, no arrowhead) ───────────────────────────────
        end = ax_vec * pole_length
        kw = dict(color=color, linewidth=2.8)
        if label:
            kw["label"] = label
        self.ax.plot([0, end[0]], [0, end[1]], [0, end[2]], **kw)

        # ── triangular pennant ─────────────────────────────────────────────
        v0 = P
        v1 = ax_vec * (pole_length - flag_width) #P + flag_len * fd + (flag_width / 2) * fd_perp
        v2 = P + flag_len * fd - (flag_width / 2) * fd_perp
        poly = Poly3DCollection(
            [np.array([v0, v1, v2])],
            alpha=alpha, facecolor=color, edgecolor=color, linewidth=0.9,
        )
        self.ax.add_collection3d(poly)

        # ── reference direction (θ = 0) ────────────────────────────────────
        if show_reference and angle > 1e-4:
            ref_tip = P + flag_len * 0.85 * u
            self.ax.plot([P[0], ref_tip[0]],
                         [P[1], ref_tip[1]],
                         [P[2], ref_tip[2]],
                         color="silver", lw=1.1, linestyle="--", alpha=0.75)

        # ── angle arc at pole tip ──────────────────────────────────────────
        if show_arc and angle > 1e-4:
            t_vals = np.linspace(0.0, angle, 48)
            arc = P[np.newaxis, :] + arc_radius * (
                np.outer(np.cos(t_vals), u) + np.outer(np.sin(t_vals), v)
            )
            self.ax.plot(arc[:, 0], arc[:, 1], arc[:, 2],
                         color=color, lw=1.8, alpha=0.85)
            # Arrowhead at arc end
            tang = arc[-1] - arc[-3]
            tn = np.linalg.norm(tang)
            if tn > 1e-14:
                tang /= tn
                self.ax.quiver(*arc[-2], *(tang * 0.05 * arc_radius),
                               color=color, lw=1.6, arrow_length_ratio=0.99)
            # Angle label at arc midpoint
            mid_t = angle / 2
            lp = P + arc_radius * 1.5 * (math.cos(mid_t) * u +
                                          math.sin(mid_t) * v)
            self.ax.text(*lp, f"{math.degrees(angle):.1f}°",
                         color=color, fontsize=9, ha="center")

        return self

    def add_unnormalized_flag(self, q: Quaternion, *,
                              color: str = "steelblue",
                              label: str | None = None,
                              flag_len: float = 0.2,
                              flag_width: float = 0.20,
                              alpha: float = 0.72,
                              show_arc: bool = True,
                              show_reference: bool = True,
                              arc_radius: float = 0.28,
                              show_norm_label: bool = True):
        """
        Draw an unnormalized quaternion as a flag whose pole length = ‖q‖.

        Encoding
        --------
        • **Pole length** = ‖q‖  — encodes the quaternion magnitude spatially.
        • **Pole direction** = rotation axis of q/‖q‖.
        • **Pennant tip angle** = θ = 2·arccos(w/‖q‖)  — encodes orientation.
        • An optional text label at the pole tip shows the numeric norm value.

        All pennant / arc / reference-line parameters are forwarded to
        :meth:`add_flag`; see that method for their descriptions.

        Parameters
        ----------
        q               : quaternion (need not be unit)
        color           : colour for pole, pennant and arc
        label           : legend label
        show_norm_label : annotate the pole tip with "‖q‖ = <value>"
        """
        n = q.norm()
        if n < 1e-14:
            raise ValueError("Cannot visualise a zero quaternion as a flag.")

        self.add_flag(q, color=color, label=label,
                      pole_length=n,
                      flag_len=flag_len,
                      flag_width=flag_width,
                      alpha=alpha,
                      show_arc=show_arc,
                      show_reference=show_reference,
                      arc_radius=arc_radius)

        if show_norm_label:
            ax_vec, _ = axis_angle(q)
            tip = ax_vec * n
            # offset slightly along the axis so the label clears the pennant
            offset = ax_vec * 0.09
            self.ax.text(*(tip + offset), f"‖q‖={n:.2f}",
                         color=color, fontsize=8, ha="center")

        return self

    def add_sandwich_rotation(self, p: Quaternion, q: Quaternion, *,
                               color_p:      str = "steelblue",
                               color_q:      str = "seagreen",
                               color_result: str = "darkorange",
                               label_p:      str | None = "p",
                               label_q:      str | None = "q",
                               label_result: str | None = "q·p·q†",
                               show_trajectory: bool = True,
                               trajectory_steps: int = 40,
                               flag_len:   float = 0.2,
                               flag_width: float = 0.18,
                               alpha:      float = 0.72):
        """
        Visualise the sandwich product  r = q · p · q†  with three flags.

        Drawn elements
        --------------
        • **p flag**  (color_p)      : the quaternion being rotated.
        • **q flag**  (color_q)      : the rotating quaternion.
        • **r flag**  (color_result) : result  r = q · p · q†.
        • **Trajectory arc** (dashed): the path swept by p's pole tip as
          q's rotation is applied incrementally.  The arc interpolates
          both direction (via SLERP on the unit part of q) and magnitude
          (linear from ‖p‖ to ‖r‖ = ‖q‖² · ‖p‖), so it connects the tip
          of the p-flag to the tip of the r-flag.

        All three flags are drawn with :meth:`add_unnormalized_flag`, so
        their pole lengths encode their respective norms.

        Parameters
        ----------
        p, q             : quaternions (need not be unit)
        color_p/q/result : colours for the three flags
        label_p/q/result : legend labels (set to None to suppress)
        show_trajectory  : draw the dashed arc from p's tip to r's tip
        trajectory_steps : number of arc segments
        flag_len/width   : pennant geometry forwarded to add_unnormalized_flag
        alpha            : pennant opacity
        """
        r = q * p * q.conj()

        # ── three flags ────────────────────────────────────────────────────
        kw = dict(flag_len=flag_len, flag_width=flag_width, alpha=alpha,
                  show_norm_label=True)
        self.add_unnormalized_flag(p, color=color_p, label=label_p, **kw)
        self.add_unnormalized_flag(q, color=color_q, label=label_q, **kw)
        self.add_unnormalized_flag(r, color=color_result, label=label_result, **kw)

        # ── trajectory arc: p's pole tip → r's pole tip ────────────────────
        if show_trajectory:
            q_unit  = _unit(q)
            p_ax, _ = axis_angle(p)
            p_norm  = p.norm()
            r_norm  = r.norm()

            pts = []
            for k in range(trajectory_steps + 1):
                t = k / trajectory_steps
                # Rotate p's axis direction by the fraction t of q's rotation
                q_t        = slerp(Quaternion.one(), q_unit, t)
                rotated_ax = rotate_vec(q_t, p_ax)
                # Scale linearly from ‖p‖ (t=0) to ‖r‖ = ‖q‖²·‖p‖ (t=1)
                scale = p_norm + t * (r_norm - p_norm)
                pts.append(rotated_ax * scale)
            pts = np.array(pts)

            self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                         color=color_result, lw=1.8,
                         linestyle="--", alpha=0.75)
            # Arrowhead at the end of the trajectory
            tang = pts[-1] - pts[-3]
            tn = np.linalg.norm(tang)
            if tn > 1e-14:
                tang /= tn
                self.ax.quiver(*pts[-2], *(tang * 0.08),
                               color=color_result, lw=1.6,
                               arrow_length_ratio=0.99)

        return self

    def add_unit_sphere(self, *, alpha: float = 0.06,
                        color: str = "lightsteelblue",
                        n: int = 30):
        """Draw a translucent unit sphere as a reference."""
        u = np.linspace(0, 2 * math.pi, n)
        v = np.linspace(0, math.pi,     n)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones(n), np.cos(v))
        self.ax.plot_surface(xs, ys, zs, alpha=alpha,
                             color=color, linewidth=0)
        return self

    # ------------------------------------------------------------------
    # Scene helpers
    # ------------------------------------------------------------------

    def draw_axes(self, length: float = 0.80, alpha: float = 0.45):
        """Draw the reference coordinate frame arrows."""
        for direction, color, lbl in [
            ([1, 0, 0], "crimson",     "$x$"),
            ([0, 1, 0], "forestgreen", "$y$"),
            ([0, 0, 1], "royalblue",   "$z$"),
        ]:
            d = np.array(direction, dtype=float) * length
            self.ax.quiver(0, 0, 0, d[0], d[1], d[2],
                           color=color, alpha=alpha,
                           linewidth=1.5, arrow_length_ratio=0.14)
            off = np.array(direction, dtype=float) * (length + 0.11)
            self.ax.text(*off, lbl, color=color, fontsize=11, ha="center")
        return self

    def set_limits(self, lim: float = 1.3):
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(-lim, lim)
        return self

    def set_view(self, elev: float, azim: float):
        self.ax.view_init(elev=elev, azim=azim)
        return self

    def legend(self, **kwargs):
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(handles, labels, **kwargs)
        return self

    def show(self, tight: bool = True):
        if tight:
            plt.tight_layout()
        plt.show()
        return self

    def save(self, path: str, dpi: int = 150, **kwargs):
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
        return self

    def close(self):
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def visualize_quaternion(q: Quaternion, *, show_conjugate: bool = True,
                         label: str = "q", **kwargs):
    """Quick single-quaternion visualisation."""
    viz = QuaternionVisualizer(**kwargs)
    viz.draw_axes().add_unit_sphere().set_limits(1.3)
    viz.add_quaternion(q, label=label, show_conjugate=show_conjugate,
                       show_original_frame=True)
    viz.legend()
    viz.show()


def visualize_flag(q: Quaternion, *, label: str = "q",
                   color: str = "steelblue", **kwargs):
    """Quick flag visualisation of a single unit quaternion."""
    viz = QuaternionVisualizer(**kwargs)
    viz.draw_axes().add_unit_sphere(alpha=0.05).set_limits(1.55)
    viz.add_flag(q, color=color, label=label)
    viz.legend()
    viz.show()


def visualize_sandwich_rotation(p: Quaternion, q: Quaternion, **kwargs):
    """Quick visualisation of the sandwich product  r = q · p · q†."""
    r = q * p * q.conj()
    norms = [p.norm(), q.norm(), r.norm()]
    lim = max(1.55, max(norms) * 1.35)
    viz = QuaternionVisualizer(
        title=f"Sandwich product  r = q · p · q†\n"
              f"‖p‖={p.norm():.2f}  ‖q‖={q.norm():.2f}  ‖r‖={r.norm():.2f}",
    )
    viz.draw_axes(length=min(0.7, lim * 0.45)).add_unit_sphere(alpha=0.04).set_limits(lim)
    viz.add_sandwich_rotation(p, q, **kwargs)
    viz.legend()
    viz.show()


def visualize_unnormalized_flag(q: Quaternion, *, label: str = "q",
                                color: str = "steelblue", **kwargs):
    """Quick flag visualisation of an unnormalized quaternion (pole length = ‖q‖)."""
    viz = QuaternionVisualizer(**kwargs)
    n = q.norm()
    lim = max(1.55, n * 1.3)
    viz.draw_axes(length=min(0.8, lim * 0.55)).add_unit_sphere(alpha=0.04).set_limits(lim)
    viz.add_unnormalized_flag(q, color=color, label=label)
    viz.legend()
    viz.show()


def visualize_slerp(q1: Quaternion, q2: Quaternion,
                    reference_vec=None, n_steps: int = 20, **kwargs):
    """Visualise the SLERP arc between two orientations."""
    viz = QuaternionVisualizer(**kwargs)
    viz.draw_axes().add_unit_sphere().set_limits(1.3)
    viz.add_slerp(q1, q2, reference_vec=reference_vec, n_steps=n_steps,
                  label_start="q₁", label_end="q₂")
    viz.legend()
    viz.show()


# ---------------------------------------------------------------------------
# Script entry point  –  python simplega/quaternion_visualization.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _from_axis_angle(axis, angle_deg):
        """Build a unit quaternion from an axis and rotation angle (degrees)."""
        angle = math.radians(angle_deg)
        ax = np.asarray(axis, dtype=float)
        ax /= np.linalg.norm(ax)
        s = math.sin(angle / 2)
        return Quaternion(math.cos(angle / 2), s * ax[0], s * ax[1], s * ax[2])

    print("Opening 7 figures — close them to exit.")
    print("  Figure 1 : q and q† (conjugate) as flags")
    print("  Figure 2 : Rotation flags: q and rotated basis vectors")
    print("  Figure 3 : SLERP shown as a sequence of flags")
    print("  Figure 4 : Composition q₁, q₂, q₁·q₂ as flags")
    print("  Figure 5 : Three arbitrary unit quaternions as flags")
    print("  Figure 6 : Unnormalized flags (pole length = ‖q‖)")
    print("  Figure 7 : Sandwich product  r = q · p · q†  (non-unit flags)")

    # ── Figure 1: q and its conjugate q† as flags ────────────────────────
    # Conjugate has the same axis, opposite angle  →  pennant flips side
    q1  = _from_axis_angle([0, 0, 1], 90)
    q1c = q1.conj()
    viz1 = QuaternionVisualizer(
        title="Flag representation: q  and  q†\n"
              "pole = rotation axis  ·  pennant angle: q=90°, q†=−90°",
        elev=28, azim=-55,
    )
    (viz1
     .draw_axes()
     .add_unit_sphere(alpha=0.05)
     .set_limits(1.55)
     .add_flag(q1,  color="steelblue", label="q  (90° ẑ)")
     .add_flag(q1c, color="tomato",    label="q† (−90° ẑ)")
     .legend(fontsize=9))

    # ── Figure 2: rotation effect — q flag + rotated basis as pure-quat flags
    # Rotating a vector v yields the pure quaternion  q·(0,v)·q†.
    # Its pole points along R(v); angle = π for all pure quaternions.
    q2 = _from_axis_angle([1, 1, 0], 120)
    basis = [
        (np.array([1., 0., 0.]), "crimson",     "R(x̂)"),
        (np.array([0., 1., 0.]), "forestgreen", "R(ŷ)"),
        (np.array([0., 0., 1.]), "royalblue",   "R(ẑ)"),
    ]
    viz2 = QuaternionVisualizer(
        title="Rotation flags: q (120° x̂+ŷ)  +  rotated basis vectors\n"
              "R(v) shown as pure-quaternion flags  (angle = π for all pure q)",
        elev=25, azim=-50,
    )
    (viz2
     .draw_axes()
     .add_unit_sphere(alpha=0.05)
     .set_limits(1.55)
     .add_flag(q2, color="seagreen", label="q (120° x̂+ŷ)"))
    for v, col, lbl in basis:
        rv = rotate_vec(q2, v)
        viz2.add_flag(Quaternion(0.0, rv[0], rv[1], rv[2]),
                      color=col, label=lbl)
    viz2.legend(fontsize=9)

    # ── Figure 3: SLERP — a sequence of flags from q₁ to q₂ ─────────────
    qa = _from_axis_angle([0, 0, 1],  45)
    qb = _from_axis_angle([1, 0, 0], 135)
    n_slerp = 6                                    # intermediate steps
    cmap = plt.matplotlib.colormaps["coolwarm"]
    viz3 = QuaternionVisualizer(
        title="SLERP flags: smooth path from q₁ (45° ẑ) to q₂ (135° x̂)\n"
              "colour fades blue → red as t goes 0 → 1",
        elev=30, azim=-60,
    )
    (viz3
     .draw_axes()
     .add_unit_sphere(alpha=0.05)
     .set_limits(1.55))
    for i, t in enumerate(np.linspace(0.0, 1.0, n_slerp + 2)):
        qi  = slerp(qa, qb, t)
        col = cmap(t)
        lbl = ("q₁" if i == 0
               else ("q₂" if i == n_slerp + 1
               else None))
        viz3.add_flag(qi, color=col, label=lbl,
                      show_reference=(i == 0 or i == n_slerp + 1))
    viz3.legend(fontsize=9)

    # ── Figure 4: composition — q₁, q₂, q₁·q₂ as flags ─────────────────
    qx = _from_axis_angle([1, 0, 0],  60)
    qz = _from_axis_angle([0, 0, 1],  90)
    qp = _unit(qx * qz)
    viz4 = QuaternionVisualizer(
        title="Composition flags: q₁ (60° x̂),  q₂ (90° ẑ),  q₁·q₂",
        elev=25, azim=-50,
    )
    (viz4
     .draw_axes()
     .add_unit_sphere(alpha=0.05)
     .set_limits(1.55)
     .add_flag(qx, color="steelblue",  label="q₁ (60° x̂)")
     .add_flag(qz, color="seagreen",   label="q₂ (90° ẑ)")
     .add_flag(qp, color="darkorange", label="q₁·q₂")
     .legend(fontsize=9))

    # ── Figure 5: three arbitrary unit quaternions as flags ───────────────
    flag_qs = [
        (_from_axis_angle([0, 0, 1],  90), "steelblue", "q₁  90° ẑ"),
        (_from_axis_angle([1, 0, 0],  60), "tomato",    "q₂  60° x̂"),
        (_from_axis_angle([1, 1, 0], 120), "seagreen",  "q₃ 120° (x̂+ŷ)"),
    ]
    viz5 = QuaternionVisualizer(
        title="Unit Quaternions as Flags\n"
              "pole = rotation axis  ·  pennant tip angle = θ",
        elev=28, azim=-45,
    )
    (viz5
     .draw_axes()
     .add_unit_sphere(alpha=0.05)
     .set_limits(1.55))
    for q_f, col_f, lbl_f in flag_qs:
        viz5.add_flag(q_f, color=col_f, label=lbl_f)
    viz5.legend(fontsize=9)

    # ── Figure 6: unnormalized flags (pole length = ‖q‖) ─────────────────
    unorm_qs = [
        (Quaternion(0.6, 0.0, 0.0, 0.6),   "steelblue",  "q₁"),   # ‖q‖ ≈ 0.85
        (Quaternion(0.7, 0.5, 0.5, 0.1),   "tomato",     "q₂"),   # ‖q‖ ≈ 0.90
        (Quaternion(1.5, 0.6, 0.0, 0.6),   "seagreen",   "q₃"),   # ‖q‖ ≈ 1.72
        (Quaternion(0.3, 0.0, 0.2, 0.0),   "darkorange", "q₄"),   # ‖q‖ ≈ 0.36
    ]
    viz6 = QuaternionVisualizer(
        title="Unnormalized Quaternions as Flags\n"
              "pole length = ‖q‖  ·  pennant tip angle = θ of q/‖q‖",
        elev=28, azim=-45,
    )
    (viz6
     .draw_axes(length=0.5)
     .add_unit_sphere(alpha=0.06)
     .set_limits(2.0))
    for q_u, col_u, lbl_u in unorm_qs:
        viz6.add_unnormalized_flag(q_u, color=col_u, label=lbl_u)
    viz6.legend(fontsize=9)

    # ── Figure 7: sandwich product  r = q · p · q† ───────────────────────
    # p: the quaternion being rotated  (‖p‖ ≈ 1.0, 60° around x)
    # q: the rotating quaternion       (‖q‖ ≈ 1.5, 90° around z, scaled)
    #p_sw = _from_axis_angle([1, 0, 0], 60)                        # unit
    p_sw = Quaternion(                                             # ‖q‖ ≈ 1.5
        1.5 * math.cos(math.radians(45)),
        1.5 * math.sin(math.radians(45)),
        0.0,
        0.0,
    )
    q_sw = Quaternion(                                             # ‖q‖ ≈ 1.5
        0.75 * math.cos(math.radians(45)),
        0.0,
        0.0,
        0.75 * math.sin(math.radians(45)),
    )
    r_sw = q_sw * p_sw * q_sw.conj()
    norms_sw = [p_sw.norm(), q_sw.norm(), r_sw.norm()]
    lim_sw = max(1.6, max(norms_sw) * 1.35)
    viz7 = QuaternionVisualizer(
        title=f"Sandwich product  r = q · p · q†\n"
              f"‖p‖={p_sw.norm():.2f}  ‖q‖={q_sw.norm():.2f}  "
              f"‖r‖={r_sw.norm():.2f} (= ‖q‖²·‖p‖)",
        elev=28, azim=-45,
    )
    (viz7
     .draw_axes(length=0.55)
     .add_unit_sphere(alpha=0.05)
     .set_limits(lim_sw)
     .add_sandwich_rotation(p_sw, q_sw)
     .legend(fontsize=9))

    plt.show()
