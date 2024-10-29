"""
This module contains functions for the implementation of the restricted 
minimum weight perfect matching decoder for a color code of any distance. 
"""
import numpy as np
import itertools 

def get_check_matrix(my_code, restricted=None): 
    '''
    Returns valid check matrix for pymatching, indices of disregarded qubits, 
    and full check matrix after restricting stabilizers to those of a choen 
    restricted lattice: red, green or blue. 
    '''
    H = my_code.stabilizers(restricted=restricted)
    ncols = my_code.n_k_d[0]
    H = H[:, ncols:] 
    H = H[~np.all(H==0, axis=1)]
    _, remaining_idx = np.unique(H, return_index=True, axis=1)

    return H.T[np.sort(remaining_idx)].T


def restrict_syndrome(my_code, syndrome, restricted):
    """Restrict syndrome data based on restricted lattice: red, green or blue."""
    return syndrome[my_code._plaquette_position_restricted[restricted]] 


def get_dict(my_code, matching_graph, syn_in_coord, syndrome): 
    """
    Return edges on a colored restricted lattice.
    """     
    syn_coord = my_code._plaquette_indices
    true_stab_ind = index_to_coord(syn_in_coord, syn_coord)
    coord_dict = dict(zip(np.arange(len(syn_in_coord)), true_stab_ind))
    coord_dict[len(syn_in_coord)] = len(syndrome) 
    edges = np.array([(coord_dict[edge[0]], coord_dict[edge[1]]) for edge in matching_graph.edges()])
    return edges


def index_to_coord(my_list, syn_coord): 
    """."""
    for i in range(len(my_list)):
        my_list[i] = syn_coord.index(my_list[i])
    return my_list


def restricted_match(my_code, m, s, syndrome, color):
    """
    Return edges of the colored restricted lattice.
    """
    syn_coord = sorted(my_code.full_syndrome_to_plaquette_indices(s, color))
    edges = get_dict(my_code, m, syn_coord, syndrome)
    return edges


def get_fault_ids(my_code, edge):
    """
    Returns fault_ids for a given edge. 
    """
    p1, p2 = edge
    plaquette_sites = my_code.plaquette_sites + [my_code.boundary_sites]
    return list(set(plaquette_sites[p1]) & set(plaquette_sites[p2]))


def systematic_errors(L):
    """
    Returns all tuples of d-on-2 errors for 4.8.8 color code of size L.
    """   
    sites = np.arange(2*L)
    err_half = []
    for k in range(int(L/2)):
        delta = k*2*(2*L-2)
        err_half.append(list(itertools.combinations(sites + delta, int(L/2))))

    return sum(err_half, [])


def logical_failures(my_code, error, H, m, color):
    """
    Returns value 0 or 1 whether logical failure has not occured or has occured
    for a given physical error based on the equality test of parity of edges crossing
    a chosen logical representative, and parity of physical errors lying on the logical.
    """
    #error = my_error_model.generate(my_code, p)[0:my_code.n_k_d[0]]
    syndrome = H@error % 2 
    s = restrict_syndrome(my_code, syndrome, color)
    edges = restricted_match(my_code, m, s, syndrome, color)
    matching_edges = edges[np.where(m.decode(s))]
    matching_edges = list(map(tuple, matching_edges))
    log_parity = np.sum(error[my_code.boundaries[3]], dtype=np.int8) % 2
    match_parity = sum(map(lambda x : x in my_code.logical_crossings, matching_edges)) % 2 

    # log_qubits = np.where(my_code.logical_xs[0][0:my_code.n_k_d[0]] == 1)[0]
    # ids = list(map(get_fault_ids, matching_edges))
    # match_parity = sum(map(lambda x : x in log_qubits, ids)) % 2
    return (log_parity + match_parity) % 2 #int(log_parity != match_parity) 


# def red_logical_fail(my_code, error, H, m, color):
#     """
#     Returns value 0 or 1 whether logical failure has not occured or has occured for a given physical error.
#     
#     """
#     syndrome = H@error % 2 
#     s = restrict_syndrome(my_code, syndrome, color)
#     edges = restricted_match(my_code, m, s, syndrome, color)
#     matching_edges = edges[np.where(m.decode(s))]
#     matching_edges = list(map(tuple, matching_edges))
#     log_flip = error[np.where(my_code.logical_xs[0][0:my_code.n_k_d[0]] == 1)]
#     log_parity = sum(map(lambda x : x == 1, log_flip)) % 2
#     match_parity = sum(map(lambda x : x in my_code.red_logical_crossings, matching_edges)) % 2 

#     # log_qubits = np.where(my_code.logical_xs[0][0:my_code.n_k_d[0]] == 1)[0]
#     # ids = list(map(get_fault_ids, matching_edges))
#     # match_parity = sum(map(lambda x : x in log_qubits, ids)) % 2
#     return (log_parity + match_parity) % 2 #int(log_parity != match_parity) 

