"""
Basis routing function.
Python port of SimpleGA.jl/src/basis.jl

Returns the list of basis vectors for the requested algebra signature (p, q, r).
"""

from . import ga20, ga30, ga31


def basis(p: int = None, q: int = 0, r: int = 0):
    """
    Return basis vectors for GA(p, q, r).

    Parameters
    ----------
    p : int
        Number of positive-signature dimensions.
    q : int, optional
        Number of negative-signature dimensions (default 0).
    r : int, optional
        Number of null dimensions (default 0).

    Returns
    -------
    list of basis Odd elements, or prints supported algebras if called with no arguments.

    Supported algebras
    ------------------
    GA(2,0)  -> ga20.basis
    GA(3,0)  -> ga30.basis
    GA(3,1)  -> ga31.basis
    """
    if p is None:
        print("Supported algebras:")
        print("  GA(2,0) -> simplega.ga20   (complex representation)")
        print("  GA(3,0) -> simplega.ga30   (quaternion representation)")
        print("  GA(3,1) -> simplega.ga31   (2x2 complex matrix representation)")
        return None

    if r != 0:
        # Degenerate algebras: promote to higher-signature non-degenerate
        return basis(p + r, q + r, 0)

    if q == 0:
        if p <= 2:
            return ga20.basis
        if p == 3:
            return ga30.basis
        raise NotImplementedError(f"GA({p},0) not implemented; use simplega.ga30 or higher.")

    if q == 1:
        if p == 3:
            return ga31.basis
        raise NotImplementedError(f"GA({p},{q}) not implemented.")

    raise NotImplementedError(
        f"GA({p},{q},{r}) not implemented in this port. "
        "See https://github.com/MonumoLtd/SimpleGA.jl for the full Julia package."
    )
