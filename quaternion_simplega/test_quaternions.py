"""
Tests for the Quaternion type.
Python port of the quaternion functionality in SimpleGA.jl/src/quaternions.jl
"""

import math
#import pytest
import sys, os
path_to_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, path_to_root)
from quaternions import Quaternion, dot


# ------------------------------------------------------------------
# Construction and identity elements
# ------------------------------------------------------------------

def test_zero():
    q = Quaternion.zero()
    assert q.w == 0.0 and q.x == 0.0 and q.y == 0.0 and q.z == 0.0


def test_one():
    q = Quaternion.one()
    assert q.w == 1.0 and q.x == 0.0 and q.y == 0.0 and q.z == 0.0


def test_norm_zero():
    assert Quaternion.zero().norm() == 0.0


def test_norm_one():
    assert Quaternion.one().norm() == 1.0


# ------------------------------------------------------------------
# Arithmetic
# ------------------------------------------------------------------

def test_negation():
    q = Quaternion(1, 2, 3, 4)
    assert -q == Quaternion(-1, -2, -3, -4)


def test_add():
    a = Quaternion(1, 2, 3, 4)
    b = Quaternion(5, 6, 7, 8)
    assert a + b == Quaternion(6, 8, 10, 12)


def test_sub():
    a = Quaternion(5, 6, 7, 8)
    b = Quaternion(1, 2, 3, 4)
    assert a - b == Quaternion(4, 4, 4, 4)


def test_scalar_add():
    q = Quaternion(1, 2, 3, 4)
    assert (q + 10) == Quaternion(11, 2, 3, 4)
    assert (10 + q) == Quaternion(11, 2, 3, 4)


def test_scalar_sub():
    q = Quaternion(5, 1, 2, 3)
    assert (q - 2) == Quaternion(3, 1, 2, 3)
    assert (10 - q) == Quaternion(5, -1, -2, -3)


def test_scalar_mul():
    q = Quaternion(1, 2, 3, 4)
    assert 2 * q == Quaternion(2, 4, 6, 8)
    assert q * 2 == Quaternion(2, 4, 6, 8)


def test_mul_identity():
    q = Quaternion(1, 2, 3, 4)
    one = Quaternion.one()
    assert q * one == q
    assert one * q == q


def test_mul_ij_k():
    i = Quaternion(0, 1, 0, 0)
    j = Quaternion(0, 0, 1, 0)
    k = Quaternion(0, 0, 0, 1)
    assert i * j == k,  f"i*j = {i*j}"
    assert j * k == i,  f"j*k = {j*k}"
    assert k * i == j,  f"k*i = {k*i}"
    # anti-commutativity
    assert j * i == -k, f"j*i = {j*i}"


def test_mul_i_squared():
    i = Quaternion(0, 1, 0, 0)
    expected = Quaternion(-1, 0, 0, 0)
    assert i * i == expected, f"i^2 = {i*i}"


def test_scalar_div():
    q = Quaternion(2, 4, 6, 8)
    assert (q / 2).isapprox(Quaternion(1, 2, 3, 4))


def test_quat_div():
    q = Quaternion(1, 2, 3, 4)
    assert (q / q).isapprox(Quaternion.one(), rtol=1e-10)


# ------------------------------------------------------------------
# Conjugate / adjoint
# ------------------------------------------------------------------

def test_conj():
    q = Quaternion(1, 2, 3, 4)
    assert q.conj() == Quaternion(1, -2, -3, -4)


def test_norm_via_conj():
    q = Quaternion(1, 2, 3, 4)
    nrm = math.sqrt(dot(q, q.conj()))
    assert math.isclose(nrm, q.norm(), rel_tol=1e-12)


# ------------------------------------------------------------------
# Dot product
# ------------------------------------------------------------------

def test_dot_self():
    i = Quaternion(0, 1, 0, 0)
    assert dot(i, i) == -1.0


def test_dot_one():
    q = Quaternion.one()
    assert dot(q, q) == 1.0


# ------------------------------------------------------------------
# Real / imag parts
# ------------------------------------------------------------------

def test_real_part():
    q = Quaternion(3, 1, 2, 4)
    assert q.real_part() == Quaternion(3, 0, 0, 0)


def test_imag_part():
    q = Quaternion(3, 1, 2, 4)
    assert q.imag_part() == Quaternion(0, 1, 2, 4)


# ------------------------------------------------------------------
# Exponential
# ------------------------------------------------------------------

def test_bivector_exp_unit():
    # exp(pi/2 * i) = cos(pi/2) + sin(pi/2) * i = j... wait: i^2 = -1
    i = Quaternion(0, 1, 0, 0)
    t = math.pi / 2
    R = (t * i).bivector_exp()
    expected = Quaternion(math.cos(t), math.sin(t), 0, 0)
    assert R.isapprox(expected, rtol=1e-12), f"bivector_exp mismatch: {R}"


def test_exp_preserves_norm():
    import random
    random.seed(5)
    q = Quaternion(0, random.random(), random.random(), random.random())
    R = q.exp()
    # exp of a pure imaginary quaternion has norm 1
    assert math.isclose(R.norm(), 1.0, rel_tol=1e-10), f"norm(exp(q)) = {R.norm()}"


# ------------------------------------------------------------------
# isapprox / equality
# ------------------------------------------------------------------

def test_isapprox_self():
    q = Quaternion(1, 2, 3, 4)
    assert q.isapprox(q)


def test_isapprox_close():
    q = Quaternion(1, 2, 3, 4)
    q2 = Quaternion(1 + 1e-12, 2, 3, 4)
    assert q.isapprox(q2, rtol=1e-9)


def test_isapprox_far():
    q = Quaternion(1, 2, 3, 4)
    q2 = Quaternion(2, 2, 3, 4)
    assert not q.isapprox(q2)
