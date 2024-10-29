import numpy as np
import itertools 
import pymatching
import scipy
from scipy.sparse import csc_matrix, csr_matrix
from typing import List, Union


def get_check_matrix(my_code, restricted=None): 
    '''
    Returns valid check matrix for pymatching, indices of disregarded qubits, and full check matrix
    '''
    H = my_code.stabilizers(restricted=restricted)
    H = H[:, my_code.n_k_d[0]:] 
    H = H[~np.all(H==0, axis=1)]
    return H 

def unified_check_matrix(H, L, my_code):
    # top boundary
    n = my_code.n_k_d[0]

    js = np.array(my_code.boundaries[0])
    H[:, js] = np.add(H[:, js], H[:, js + 2*n - L])

    js = np.array(my_code.boundaries[1])
    H[:,js] = np.add(H[:,js], H[:,js+L])

    js = np.array(my_code.boundaries[2])
    H[:,js] = np.add(H[:,js], H[:,js + 2*n + (L-1)])

    js = np.array(my_code.boundaries[3])
    H[:,js] = np.add(H[:,js], H[:,js + 2*n - (L-1)])
    H = np.delete(H, np.add(my_code.boundaries[0], n).tolist() + np.add(my_code.boundaries[1],n).tolist()+ np.add(my_code.boundaries[2],2*n).tolist() + np.add(my_code.boundaries[3],2*n).tolist()
    , axis=1) 
    H[H > 0] = 1
    return H

def restrict_unified_check_matrix(H):
    """
    .
    """
    _, remaining_idx = np.unique(H, return_index=True, axis=1)
    return H.T[np.sort(remaining_idx)].T, remaining_idx


def get_fault_ids(check_mat, edge):
    p, q = edge
    arr1 = np.where(check_mat[p]==1)[0]
    arr2 = np.where(check_mat[q]==1)[0]   
    return np.intersect1d(arr1, arr2)


def restrict_syndrome(syndrome, restricted, my_code):
    return syndrome[my_code._plaquette_position_restricted[restricted]].tolist() 


def expand_check_matrix(H_red, H_blue, H_green, my_code):
    
    """Merge the check matrice of the restricted lattices."""
    check_matrix = np.zeros( (H_red.shape[0] + 2*H_blue.shape[0], 3*my_code.n_k_d[0] ) )
    check_matrix[:H_red.shape[0], :my_code.n_k_d[0]] = H_red
    check_matrix[H_red.shape[0]:H_red.shape[0]+H_blue.shape[0], my_code.n_k_d[0]:2*my_code.n_k_d[0]] = H_blue
    check_matrix[H_red.shape[0]+H_blue.shape[0]:H_red.shape[0]+2*H_blue.shape[0], 2*my_code.n_k_d[0]:3*my_code.n_k_d[0]] = H_green
    
    return check_matrix

def reflect(my_code, syndrome):
    test = [i for i in np.array_split(syndrome, my_code.n_k_d[2]-1)]
    return np.hstack(test[::-1])

def get_unified_syndrome(syndrome, my_code):
    s_red = restrict_syndrome(syndrome, 'red', my_code)
    s_blue = restrict_syndrome(reflect(my_code, syndrome), 'blue', my_code)
    s_green = restrict_syndrome(np.hstack([i[::-1] for i in np.array_split(syndrome, my_code.n_k_d[2]-1)]), 'green', my_code)
    return s_red + s_blue + s_green


def systematic_errors(L):
    """
    Returns tuples of ALL d-on-2 errors for 4.8.8 color code of size L.
    """   
    sites = np.arange(2*L)
    err_half = []
    for k in range(int(L/2)):
        delta = k*2*(2*L-2)
        err_half.append(list(itertools.combinations(sites + delta, int(L/2))))
    return sum(err_half, [])


def systematic_errors_row(L, k):
    """
    Returns tuples of ALL d-on-2 errors on row k for 4.8.8 color code of size L.
    """   
    d = k*2*(2*L-2)
    sites = np.arange(2*L) + d
    return np.array(list(itertools.combinations(sites, int(L/2))))


def systematic_correctable_errors(L):
    
    errors = []
    for k in range(int(L/2)): 
        d = k*2*(2*L-2) 
        sites = np.arange(2*L) + d
        for i in np.arange(0,int(L),2): 
            rest = list(set(sites) - set([i+d, i+1+d, i+L+d, i+L+1+d]))
            for j in rest:
                errors.append([i+d, i+L+1+d, j])
                errors.append([i+1+d, i+L+d, j])
    return errors


def systematic_correctable_errors_row(L, k):
    """
    Correctable d/2-errors by the unified lattice
    """
    errors = []
    d = k*2*(2*L-2) 
    sites = np.arange(2*L) + d
    for i in np.arange(0,int(L),2): # 0, 2, 4 squares
        rest = list(set(sites) - set([i+d, i+1+d, i+L+d, i+L+1+d]))
        if L == 6:
            for j in rest:
                errors.append([i+d, i+L+1+d] + [j])
                errors.append([i+1+d, i+L+d] + [j])
        else:
            rest_combinations = [list(i) for i in list(itertools.combinations(rest, int(L/2) - 2))]
            for j in rest_combinations:
                errors.append([i+d, i+L+1+d] + j)
                errors.append([i+1+d, i+L+d] + j)
    return errors

# < Various heuristics functions assigning weights to edges on the unified matching graph > :

def weight_tailored_at_boundaries(my_code, ids):
    '''
    Return weight of edge in lattice L depending on its position
    '''
    rg_sides = list(set(my_code.boundaries[0] + my_code.boundaries[1]) - set(my_code.corners))
    rb_sides = list(set(my_code.boundaries[2] + my_code.boundaries[3]) - set(my_code.corners))
    
    if any(id in my_code.corners for id in ids):
        weight = 2
    elif any(id in rb_sides for id in ids):
        weight = 2
    elif any(id in rg_sides for id in ids):
        weight = 1
    else:
        weight = 1
    return weight 


def assign_weights(my_code, ids):
    '''
    Return weight of edge in lattice L depending on its position.
    '''
    sides = list(set(my_code.boundary_sites) - set(my_code.corners))
    if any(id in my_code.corners for id in ids):
        weight = 3
    elif any(id in sides for id in ids):
        weight = 2
    else:
        weight = 1
    return weight 


def logical_failures(my_code, error, syndrome, cross, m):
    """
    Returns the number of failures, i.e number of edges crossing the logical X operator defined in my_code.logical_xs
    """ 
    unified_syndrome = get_unified_syndrome(syndrome, my_code)
    matching = m.decode(unified_syndrome)
    matching_ind = np.where(matching)
    es = m.edges()
    matching_edges = [es[i][0:2] for i in matching_ind[0]]
    logical_X = my_code.boundaries[3] 
    log_parity = np.sum(error[logical_X]) % 2
    match_parity = sum(map(lambda x : x in cross, matching_edges)) % 2
    
    return (log_parity + match_parity) % 2 


def get_unified_syndrome_matrix_M(my_code, matching_graph: pymatching.Matching, num_stabilisers):
    M = np.zeros((matching_graph.num_detectors, num_stabilisers), dtype=np.uint8)
    for i in range(num_stabilisers):
        s = np.zeros(num_stabilisers, dtype=np.uint8)
        s[i] = 1
        M[:, i] =  get_unified_syndrome(s, my_code)
    return csr_matrix(M)


def predict_logical_flip(syndrome, m: pymatching.Matching, M: csr_matrix) -> np.ndarray:
    """
    Returns whether or not the logical observable(s) were flipped.
    E.g. if only there is only a single logical observable, then 
    the pymatching.Matching graph should have a single fault id, 
    e.g. should have `m.num_fault_ids == 1`
    """ 
    unified_syndrome = M@syndrome % 2
    # assert False
    return m.decode(unified_syndrome)

def predict_edge_flips(syndrome, m: pymatching.Matching, M: csr_matrix):
    unified_syndrome = M@syndrome % 2
    # assert False
    return m.decode(unified_syndrome)


def load_from_check_matrix_with_fault_ids(
                            H: Union[scipy.sparse.spmatrix, np.ndarray, List[List[int]]],
                            spacelike_weights: Union[float, np.ndarray, List[float]] = None,
                            fault_ids: csc_matrix = None,
                            merge_strategy: str = "smallest-weight",
                            **kwargs
                            ) -> None:
    """
    Load a matching graph from a check matrix
    Parameters
    ----------
    H : `scipy.spmatrix` or `numpy.ndarray` or List[List[int]]
        The quantum code to be decoded with minimum-weight perfect
        matching, given as a binary check matrix (scipy sparse
        matrix or numpy.ndarray)
    spacelike_weights : float or numpy.ndarray, optional
        If `H` is given as a scipy or numpy array, `spacelike_weights` gives the weights
        of edges in the matching graph corresponding to columns of `H`.
        If spacelike_weights is a numpy.ndarray, it should be a 1D array with length
        equal to `H.shape[1]`. If spacelike_weights is a float, it is used as the weight for all
        edges corresponding to columns of `H`. By default None, in which case
        all weights are set to 1.0
    error_probabilities : float or numpy.ndarray, optional
        The probabilities with which an error occurs on each edge associated with a
        column of H. If a
        single float is given, the same error probability is used for each
        column. If a numpy.ndarray of floats is given, it must have a
        length equal to the number of columns in H. This parameter is only
        needed for the Matching.add_noise method, and not for decoding.
        By default None
    repetitions : int, optional
        The number of times the stabiliser measurements are repeated, if
        the measurements are noisy. By default None
    timelike_weights : float or numpy.ndarray, optional
        If `repetitions>1`, `timelike_weights` gives the weight of
        timelike edges. If a float is given, all timelike edges weights are set to
        the same value. If a numpy array of size `(H.shape[0],)` is given, the
        edge weight for each vertical timelike edge associated with the `i`th check (row)
        of `H` is set to `timelike_weights[i]`. By default None, in which case all
        timelike weights are set to 1.0
    measurement_error_probabilities : float or numpy.ndarray, optional
        If `repetitions>1`, gives the probability of a measurement
        error to be used for the add_noise method. If a float is given, all measurement
        errors are set to the same value. If a numpy array of size `(H.shape[0],)` is given,
        the error probability for each vertical timelike edge associated with the `i`th check
        (row) of `H` is set to `measurement_error_probabilities[i]`. This argument can also be
        given using the keyword argument `measurement_error_probability` to maintain backward
        compatibility with previous versions of Pymatching. By default None
    Examples
    --------
    >>> import pymatching
    >>> m = pymatching.Matching([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
    >>> m
    <pymatching.Matching object with 3 detectors, 1 boundary node, and 4 edges>
    Matching objects can also be initialised from a sparse scipy matrix:
    >>> import pymatching
    >>> from scipy.sparse import csc_matrix
    >>> H = csc_matrix([[1, 1, 0], [0, 1, 1]])
    >>> m = pymatching.Matching(H)
    >>> m
    <pymatching.Matching object with 2 detectors, 1 boundary node, and 3 edges>
    """
    try:
        H = csc_matrix(H)
    except TypeError:
        raise TypeError("H must be convertible to a scipy.csc_matrix")
    H = H.astype(np.uint8)
    num_edges = H.shape[1]
    weights = 1.0 if spacelike_weights is None else spacelike_weights
    if isinstance(weights, (int, float, np.integer, np.floating)):
        weights = np.ones(num_edges, dtype=float)*weights
    weights = np.asarray(weights)


    H.eliminate_zeros()
    fault_ids = csc_matrix(fault_ids)
    fault_ids.eliminate_zeros()
    g = pymatching.Matching()
    for i in range(len(H.indptr) - 1):
        s, e = H.indptr[i:i + 2]
        v1 = H.indices[s]
        fid_idx1, fid_idx2 = fault_ids.indptr[i:i+2]
        fault_ids_indices = set(int(fault_ids.indices[j]) for j in range(fid_idx1, fid_idx2))
        if e - s == 1:
            g.add_boundary_edge(v1, fault_ids=fault_ids_indices, weight=weights[i], merge_strategy=merge_strategy)
        else:
            v2 = H.indices[e - 1]
            g.add_edge(v1, v2, fault_ids=fault_ids_indices, weight=weights[i], merge_strategy=merge_strategy)
    return g


def print_load_from_check_matrix_with_fault_ids():
    """
    Attempt at using latest pymatching version.
    """
    H = csc_matrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
    fault_ids = csc_matrix([[0, 1, 0, 0], [0, 1, 1, 0]])
    m = load_from_check_matrix_with_fault_ids(H=H, fault_ids=fault_ids)
    print(m.edges())


def test_load_from_check_matrix_with_fault_ids():
    """
    Testing latest pymatching version.
    """
    H = csc_matrix([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    fault_ids = csc_matrix([[0, 1, 0]])
    m = load_from_check_matrix_with_fault_ids(H=H, fault_ids=fault_ids)
    assert m.edges() == [(0, 2, {'fault_ids': set(), 'weight': 1.0, 'error_probability': -1.0}), (0, 1, {'fault_ids': {0}, 'weight': 1.0, 'error_probability': -1.0}), (1, 2, {'fault_ids': set(), 'weight': 1.0, 'error_probability': -1.0})]


# weights as split by A and B parameters

# def test_weights(my_code, ids, weights):
#     '''
#     Return weight of edge in lattice L depending on its position.
#     '''
#     A, B = weights
#     # Corner weights set to A+B
#     if my_code._site_label[ids[0]] == 'corner':
#         weight = A+B # from 2*A + B

#     # GB to RB boundary weights set to A+B
#     elif my_code._site_label[ids[0]] == 'rb boundary':
#         weight = A+B

#     elif my_code._site_label[ids[0]] == 'rg boundary':
#         weight = A+B

#     # GB bulk
#     elif my_code._site_label[ids[0]] == 'gb bulk':
#         weight = B

#     # RG and RB bulk
#     else:
#         weight = A
        
#     return weight


def test_weights_config_c(my_code, ids, weights):
    '''
    Return weight of edge in lattice L depending on its position.
    '''
    A, B = weights
    # Corner weights set to A+B
    if my_code._site_label[ids[0]] == 'corner':
        weight = A+B # from 2*A + B

    # GB to RB boundary weights set to A+B
    elif my_code._site_label[ids[0]] == 'rb boundary':
        weight = A+B

    elif my_code._site_label[ids[0]] == 'rg boundary':
        weight = B

    # GB bulk
    elif my_code._site_label[ids[0]] == 'gb bulk':
        weight = B

    # RG and RB bulk
    else:
        weight = A
        
    return weight


if __name__ == "__main__":
    print_load_from_check_matrix_with_fault_ids()