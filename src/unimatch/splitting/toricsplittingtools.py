import numpy as np
import random
from collections import Counter
from pymatching import Matching
import unimatch.decoders.decodertools as rdt
import unimatch.decoders.unifiedColor488Decoder as udt 
from unimatch.codes.color import Color488Code

def g_func(x):
    """ 
    Detailed balance function.
    """
    return 1/(1+x)

def square_pairs(sites):
    return [sites[:2], sites[::2], sites[1:3]] 
    
def string_p(p, n_sites, error):
    x = (p/3)**(len(error)/2) * (1-p)**(n_sites - len(error))
    return x 

def metropolis_samples(p_j, err_0, n_samples, my_code, H_full, m, decoder, H_unified=None):
    """
    Metropolis routine in the plitting method https://link.aps.org/doi/10.1103/PhysRevA.88.062308. 
    Returns a list of uncorrectable error configurations drawn from probability distribution Pi_j. 
    """
    e_samples = []
    if H_unified is not None:
        logical_crossings = [tuple(np.where(H_unified[:,i])[0]) for i in my_code.boundaries[3]] # for unified 

    for _ in range(n_samples):

        # Pick a random two-qubit on red square configuration uniformly over the entire code.
        
        sq = random.choice(my_code._red_plaquette_indices)
        sq_sites = random.choice(square_pairs(my_code.plaquette_sites[sq]))
        error_test = [item for item, count in Counter(np.array(err_0 + sq_sites)).items() if count % 2]
        full_error_0, full_error_test = np.zeros(my_code.n_k_d[0]), np.zeros(my_code.n_k_d[0])
        full_error_0[err_0], full_error_test[error_test] = 1, 1

        # test new error for failure 
        n_sites = len(my_code._red_plaquette_indices) 
        q = np.amin(np.array([1, string_p(p_j, n_sites, error_test) / string_p(p_j, n_sites, err_0)])) 

        b = np.random.choice([0, 1], p=[1-q, q])

        if decoder == 'restricted':
            if b == 0:
                e_samples.append(err_0)

            elif b == 1 and rdt.logical_failures(my_code, full_error_test, H_full, m, 'green') == 1:
                e_samples.append(error_test)
                err_0 = error_test
            else:
                e_samples.append(err_0)
        
        elif decoder == 'unified':
            syndrome = H_full@full_error_test % 2 
            if b == 0:
                e_samples.append(err_0)

            elif b == 1 and udt.logical_failures(my_code, full_error_test, syndrome, logical_crossings, m) == 1:

                e_samples.append(error_test)
                err_0 = error_test

            else:
                e_samples.append(err_0)

    return e_samples


def physical_error_splits(p_0, t, my_code):  
    """
    Returns the array of t physical error probabilities starting from p_0 
    """
    ps = np.zeros(t); ps[0] = p_0

    for i in range(t-1):
        n_sites = len(my_code._red_plaquette_indices) 
        # ps[i+1] = ps[i]*2**(-1/np.sqrt(max(n/2, ps[i]*n/2)))
        ps[i+1] = ps[i]*2**(-1/np.sqrt(max(my_code.n_k_d[2]/2, ps[i]*n_sites/2)))

    return ps

def rescale(arr):
    return arr[int(len(arr)/2):]

def generate_splitting_data_restricted(L : np.uint8, p_list : np.ndarray, P_0, err_0 : np.ndarray, c_max : np.float32, nsamples : np.uint8):
    """
    Returns an analytical estimate array, and a numerical estimate from the splitting method, estimated using nsamples Metropolis samples.
    L : size of the lattice
    p_array : 
    """
    my_code = Color488Code(L)
    H_full = my_code.stabilizers(restricted=None)[:, my_code.n_k_d[0]:]
    H_green = rdt.get_check_matrix(my_code, 'green')
    m = Matching(H_green)

    error_chains = []
    
    for p in p_list:
        error_chains.append((rescale(metropolis_samples(p, err_0, nsamples, my_code, H_full, m, 'restricted', H_unified=None))))  

    n_sites = len(my_code._red_plaquette_indices) 
    n_splits = len(p_list) - 1 
    c_list = np.zeros(n_splits)
    c_vals = np.linspace(0.001, c_max, 100)

    for i in range(n_splits):
        avg_top, avg_bottom = np.zeros(len(c_vals)), np.zeros(len(c_vals))
        for i_c, c in enumerate(c_vals):

            avg_top[i_c] = sum(np.array([g_func((c)*string_p(p_list[i], n_sites, j)/string_p(p_list[i+1], n_sites, j)) for j in error_chains[i] ])) / len(error_chains[i])
            avg_bottom[i_c] = sum(np.array([g_func((1/c)*string_p(p_list[i+1], n_sites, j)/string_p(p_list[i], n_sites, j)) for j in error_chains[i+1] ])) / len(error_chains[i+1])
        
        divided_avg =  avg_top / avg_bottom
        idx = np.argmin(np.abs(avg_top - avg_bottom))
        c_list[i] = c_vals[idx] * divided_avg[idx]    

    plogs = np.zeros(n_splits+1); plogs[0] = P_0  

    for k in range(n_splits):
        plogs[k+1] = np.prod(c_list[:k+1]) * plogs[0]

    return plogs


def weight_func(my_code, ids, weights):
    '''
    Return weight of edge in lattice L depending on its position.
    '''
    L = my_code.size
    A, B = weights
    N = my_code.n_k_d[0] + my_code.size + int(my_code.size/2 - 1)*2*(my_code.size-2) + int(my_code.size/2 - 2)*2*my_code.size
    A1 = 2*(my_code.n_k_d[0]-L)
    B1 = 4*my_code.size - 8

    # Corner weights set to A+B
    if my_code._site_label[ids[0]] == 'corner':
        weight = A+B 
    # GB to RB boundary weights set to A+B
    elif my_code._site_label[ids[0]] == 'rg boundary':
        weight = B 
    elif my_code._site_label[ids[0]] == 'rb boundary':
        weight = A+B
    # GB bulk
    elif my_code._site_label[ids[0]] == 'gb bulk':
        weight = B 
    # RG and RB bulk
    elif my_code._site_label[ids[0]] == 'rb bulk':
        weight = A 
    elif my_code._site_label[ids[0]] == 'rg bulk' and ids[0] in range(my_code.n_k_d[0], my_code.n_k_d[0]+my_code.size):
        weight = 0
    elif my_code._site_label[ids[0]] == 'rg bulk' and ids[0] in range(N, N+my_code.size):  
        weight = 0
    else:
        weight = B
    return weight 


def generate_splitting_data_unified(L, p_list, P_0, err_0, c_max, nsamples, B):
    """
    Returns an analytical estimate array, and a numerical estimate from the splitting method, estimated using nsamples Metropolis samples.
    L : size of the lattice
    p_array : 
    """
    # 1. Initialise code characteristics

    my_code = Color488Code(L)
    H_full = my_code.stabilizers(restricted=None)[:, my_code.n_k_d[0]:]
    H_green = udt.get_check_matrix(my_code, 'green')
    H_blue = udt.get_check_matrix(my_code, 'blue')
    H_red = udt.get_check_matrix(my_code, 'red')
    check_matrix = udt.expand_check_matrix(H_red, H_blue, H_green, my_code)
    H_unified = udt.unified_check_matrix(check_matrix, L, my_code)
    H_graph, _ = udt.restrict_unified_check_matrix(H_unified)
    m = Matching(H_graph)
    # weights = [udt.test_weights_config_c(my_code, udt.get_fault_ids(H_unified, (e[0],e[1])), [1, B]) for e in m.edges()]
    weights = [weight_func(my_code, udt.get_fault_ids(H_unified, (e[0],e[1])), [1, B]) for e in m.edges()]
    # weights = [udt.weight_tailored_at_boundaries(my_code, udt.get_fault_ids(H_unified, (e[0],e[1]))) for e in m.edges()]

    m_with_weights = Matching()
    m_with_weights.load_from_check_matrix(H_graph, spacelike_weights=weights)

    error_chains = []

    for p in p_list:
        #err_start = list(err_0[np.random.choice(range(len(err_0)))])
        error_chains.append((rescale(metropolis_samples(p, err_0, nsamples, my_code, H_full, m_with_weights, 'unified', H_unified=H_unified))))  

    n_sites = len(my_code._red_plaquette_indices) 
    n_splits = len(p_list) - 1
    c_list = np.zeros(n_splits)
    c_vals = np.linspace(0.001, c_max, 100)

    for i in range(n_splits):
        avg_top, avg_bottom = np.zeros(len(c_vals)), np.zeros(len(c_vals))
        for i_c, c in enumerate(c_vals):
            avg_top[i_c] = sum(np.array([g_func((c)*string_p(p_list[i], n_sites, j)/string_p(p_list[i+1], n_sites, j)) for j in error_chains[i] ])) / len(error_chains[i])
            avg_bottom[i_c] = sum(np.array([g_func((1/c)*string_p(p_list[i+1], n_sites, j)/string_p(p_list[i], n_sites, j)) for j in error_chains[i+1] ])) / len(error_chains[i+1])
        
        divided_avg =  avg_top / avg_bottom
        idx = np.argmin(np.abs(avg_top - avg_bottom))
        c_list[i] = c_vals[idx] * divided_avg[idx]    

    plogs = np.zeros(n_splits+1); plogs[0] = P_0

    for k in range(n_splits):
        plogs[k+1] = np.prod(c_list[:k+1]) * plogs[0]

    return plogs


