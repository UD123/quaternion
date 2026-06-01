"""
Quaternion implementation.
Python port of SimpleGA.jl/src/quaternions.jl

A Quaternion is a number of the form  w + x*i + y*j + z*k
with the standard quaternion multiplication rules.
"""

import math


class Quaternion:
    """
    Quaternion division algebra element.

    Fields: w (real), x, y, z (imaginary).
    """

    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w=0.0, x=0.0, y=0.0, z=0.0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # ------------------------------------------------------------------
    # Class helpers
    # ------------------------------------------------------------------
    @classmethod
    def zero(cls):
        return cls(0.0, 0.0, 0.0, 0.0)

    @classmethod
    def one(cls):
        return cls(1.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def __neg__(self):
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.w + other.w, self.x + other.x,
                              self.y + other.y, self.z + other.z)
        if isinstance(other, (int, float)):
            return Quaternion(self.w + other, self.x, self.y, self.z)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(self.w + other, self.x, self.y, self.z)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.w - other.w, self.x - other.x,
                              self.y - other.y, self.z - other.z)
        if isinstance(other, (int, float)):
            return Quaternion(self.w - other, self.x, self.y, self.z)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(other - self.w, -self.x, -self.y, -self.z)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(other * self.w, other * self.x,
                              other * self.y, other * self.z)
        if isinstance(other, Quaternion):
            a, b = self, other
            return Quaternion(
                a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
                a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
                a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
            )
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(other * self.w, other * self.x,
                              other * self.y, other * self.z)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            inv = 1.0 / other
            return inv * self
        if isinstance(other, Quaternion):
            return self * other.conj() / dot(other, other.conj())
        return NotImplemented

    # ------------------------------------------------------------------
    # Conjugate / adjoint (reverse)
    # ------------------------------------------------------------------
    def conj(self):
        """Quaternion conjugate: w - xi - yj - zk."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    @property
    def adjoint(self):
        """Alias for conj (reverse)."""
        return self.conj()

    # ------------------------------------------------------------------
    # Projections
    # ------------------------------------------------------------------
    def real_part(self):
        """Scalar (real) part as a Quaternion."""
        return Quaternion(self.w, 0.0, 0.0, 0.0)

    def imag_part(self):
        """Vector (imaginary) part as a Quaternion."""
        return Quaternion(0.0, self.x, self.y, self.z)

    def tr(self):
        """Scalar trace."""
        return self.w

    def dot(self, other):
        """Quaternion inner product: w*w - x*x - y*y - z*z."""
        if isinstance(other, Quaternion):
            return (self.w * other.w - self.x * other.x
                    - self.y * other.y - self.z * other.z)
        return NotImplemented

    def norm(self):
        """Euclidean norm."""
        return math.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)

    # ------------------------------------------------------------------
    # Exponential
    # ------------------------------------------------------------------
    def bivector_exp(self):
        """Exponential of the imaginary (bivector) part only: exp(imag(self))."""
        a = self.imag_part()
        nrm = a.norm()
        if nrm == 0.0:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        return math.cos(nrm) + math.sin(nrm) / nrm * a

    def exp(self):
        """Full quaternion exponential: exp(w) * bivector_exp(self)."""
        R = self.bivector_exp()
        return R if self.w == 0.0 else math.exp(self.w) * R

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    def isapprox(self, other, rtol: float = 1e-9, atol: float = 0.0) -> bool:
        if not isinstance(other, Quaternion):
            return False
        return (math.isclose(self.w, other.w, rel_tol=rtol, abs_tol=atol) and
                math.isclose(self.x, other.x, rel_tol=rtol, abs_tol=atol) and
                math.isclose(self.y, other.y, rel_tol=rtol, abs_tol=atol) and
                math.isclose(self.z, other.z, rel_tol=rtol, abs_tol=atol))

    def __eq__(self, other):
        if isinstance(other, Quaternion):
            return (self.w == other.w and self.x == other.x
                    and self.y == other.y and self.z == other.z)
        return NotImplemented

    def __repr__(self):
        return f"{self.w} + {self.x}i + {self.y}j + {self.z}k"


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------
def dot(a: Quaternion, b: Quaternion) -> float:
    """Quaternion inner product."""
    return a.dot(b)
