"""
Functions for Pauli strings and binary symplectic vectors/matrices 
borrowed from David Tuckett's qecsim.
"""

import numpy as np

def pauli_to_bsf(pauli):
    """
    Convert the given Pauli operator(s) to binary symplectic form.
    XIZIY -> (1 0 0 0 1 | 0 0 1 0 1)
    Assumptions:
    * pauli is a string of I, X, Y, Z such as 'XIZIY' or a list of such strings of the same length.
    """

    def _to_bsf(p):
        ps = np.array(list(p))
        xs = (ps == 'X') + (ps == 'Y')
        zs = (ps == 'Z') + (ps == 'Y')
        return np.hstack((xs, zs)).astype(int)

    if isinstance(pauli, str):
        return _to_bsf(pauli)
    else:
        return np.vstack([_to_bsf(p) for p in pauli])


def pauli_wt(pauli):
    """
    Return weight of given Pauli operator(s).
    """

    def _wt(p):
        return p.count('X') + p.count('Y') + p.count('Z')

    if isinstance(pauli, str):
        return _wt(pauli)
    else:
        return sum(_wt(p) for p in pauli)


def bsf_to_pauli(bsf):
    """
    Convert the given binary symplectic form to Pauli operator(s).
    (1 0 0 0 1 | 0 0 1 0 1) -> XIZIY
    """
    assert np.array_equal(bsf % 2, bsf), 'BSF {} is not in binary form'.format(bsf)

    def _to_pauli(b, t=str.maketrans('0123', 'IXZY')):  # noqa: B008 (deliberately reuse t)
        xs, zs = np.hsplit(b, 2)
        ps = (xs + zs * 2).astype(str)  # 0=I, 1=X, 2=Z, 3=Y
        return ''.join(ps).translate(t)

    if bsf.ndim == 1:
        return _to_pauli(bsf)
    else:
        return [_to_pauli(b) for b in bsf]


def bsf_wt(bsf):
    """
    Return weight of given binary symplectic form.
    """
    assert np.array_equal(bsf % 2, bsf), 'BSF {} is not in binary form'.format(bsf)
    return np.count_nonzero(sum(np.hsplit(bsf, 2)))


def bsp(a, b):
    """
    Return the binary symplectic product of A with B.
    """
    assert np.array_equal(a % 2, a), 'BSF {} is not in binary form'.format(a)
    assert np.array_equal(b % 2, b), 'BSF {} is not in binary form'.format(b)
    # let A = (A1|A2) and B = (B1|B2) return (A2|A1).(B1|B2)
    a1, a2 = np.hsplit(a, 2)
    return np.hstack((a2, a1)).dot(b) % 2
