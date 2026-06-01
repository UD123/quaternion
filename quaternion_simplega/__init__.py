"""
SimpleGA - Python port of SimpleGA.jl
Geometric Algebra implementations for GA(2,0), GA(3,0), GA(3,1), and Quaternions.

Supported algebras:
    GA(2,0)  ->  simplega.ga20   (complex number representation)
    GA(3,0)  ->  simplega.ga30   (quaternion representation)
    GA(3,1)  ->  simplega.ga31   (2x2 complex matrix representation)
    Quaternions -> simplega.quaternions

Common functions:
    project(a, n)          - extract grade-n component
    bivector_exp(a)        - exponential of bivector part
    inject(coeffs, basis)  - linear combination of basis elements
    dot(a, b)              - inner product
    tr(a)                  - scalar trace
    norm(a)                - geometric algebra norm
    adjoint(a)             - reverse (grade-conjugation)
    isapprox(a, b, ...)    - approximate equality
"""

from . import ga20, ga30, ga31, quaternions
from .basis import basis


def project(a, n: int):
    """Extract grade-n component from multivector a."""
    return a.project(n)


def bivector_exp(a):
    """Compute exponential of bivector (grade-2) part of a."""
    return a.bivector_exp()


def inject(coeffs, basis_elems):
    """Form a linear combination: sum(c_i * b_i) over (coeffs, basis_elems)."""
    result = None
    for c, b in zip(coeffs, basis_elems):
        term = float(c) * b
        result = term if result is None else result + term
    return result


def dot(a, b):
    """Geometric algebra inner product."""
    return a.dot(b)


def tr(a):
    """Scalar part (trace) of a multivector."""
    return a.tr()


def norm(a):
    """Geometric algebra norm: sqrt(|dot(a, a)|)."""
    return a.norm()


def adjoint(a):
    """Reverse (grade-reversal / Hermitian adjoint) of a multivector."""
    return a.adjoint


def isapprox(a, b, rtol: float = 1e-9, atol: float = 0.0) -> bool:
    """Approximate equality between multivectors, scalars, or lists/arrays thereof."""
    import math
    # Scalar (float/int/complex) comparison
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    import cmath
    if isinstance(a, complex) and isinstance(b, complex):
        return cmath.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    # Cross-type Even/Odd comparisons are always False
    if type(a) != type(b):
        return False
    if hasattr(a, '__len__'):
        if len(a) != len(b):
            return False
        return all(isapprox(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b))
    return a.isapprox(b, rtol=rtol, atol=atol)
