import numpy as np
from unimatch.codes import BitFlipErrorModel
from unimatch.codes.color import Color488Code
from unimatch.decoders import decodertools as dt
from unimatch.decoders import unifiedColor488Decoder as udt
from pymatching import Matching
import pandas as pd
import argparse

my_error_model = BitFlipErrorModel()

def test_weights(my_code, ids, weights):
    """
    Return weight of edge in lattice L depending on its position.
    """
    A, B = weights
    # Corner weights 
    if my_code._site_label[ids[0]] == 'corner':
        weight = A+B
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


def get_initial_splits(n_runs, decoder, output_file, weight_A=None, weight_B=None):

    """
    Run unified decoder to obtain a first data point using MC sampling
    as input point for the splitting method sampling for the unified decoder.
    """ 

    L_list = [6, 8, 10]
    p_list = [0.05] 

    results = []
    
    if decoder == "restricted":

        for L in L_list:
            my_code = Color488Code(L)
            H_full = my_code.stabilizers(restricted=None)[:, my_code.n_k_d[0]:]
            H_green = dt.get_check_matrix(my_code, 'green')
            m = Matching(H_green)

            for p in p_list:

                n_fails = 0
                for _ in range(n_runs):
                    error = my_error_model.generate(my_code, p)[0:my_code.n_k_d[0]]
                    n_fails += dt.logical_failures(my_code, error, H_full, m, 'green')

                results.append({'L':L, 'p':p, 'n_runs':n_runs, 'n_fails':n_fails, 'decoder':'restricted'})

    elif decoder == 'unified':

        for L in L_list:
            my_code = Color488Code(L)
            H_full = my_code.stabilizers(restricted=None)[:, my_code.n_k_d[0]:]
            H_green = udt.get_check_matrix(my_code, 'green')
            H_blue = udt.get_check_matrix(my_code, 'blue')
            H_red = udt.get_check_matrix(my_code, 'red')
            check_matrix = udt.expand_check_matrix(H_red, H_blue, H_green, my_code)
            H_unified = udt.unified_check_matrix(check_matrix, L, my_code)
            H_graph, _ = udt.restrict_unified_check_matrix(H_unified)
            m = Matching(H_graph)
            # weights = [test_weights(my_code, udt.get_fault_ids(H_unified, (e[0],e[1])), [weight_A, weight_B]) for e in m.edges()]
            # Alternatively:
            weights = [udt.weight_tailored_at_boundaries(my_code, udt.get_fault_ids(H_unified, (e[0],e[1]))) for e in m.edges()]
            m_with_weights = Matching()
            m_with_weights.load_from_check_matrix(H_graph, spacelike_weights=weights)
            logical_crossings = [tuple(np.where(H_unified[:,i])[0]) for i in my_code.boundaries[3]]
            
            for p in p_list:   

                n_fails = 0

                for _ in range(n_runs):
                    error = my_error_model.generate(my_code, p)[0:my_code.n_k_d[0]]
                    syndrome = H_full@error % 2 
                    n_fails += udt.logical_failures(my_code, error, syndrome, logical_crossings, m_with_weights)
        
                results.append({'L':L, 'p':p, 'n_runs':n_runs, 'n_fails':n_fails, 'decoder':'unified'})

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)


def run_threshold():
    parser = argparse.ArgumentParser(description='Calculate unified threshold')
    parser.add_argument('--n_runs', '-n', type=int, default=1000,
                        help='Set number of runs per script') 
    parser.add_argument('--decoder', '-dc', type=str,
                    help='which decoder')
    parser.add_argument('--output_file', '-o', type=str,
                    help='output file')
    args = parser.parse_args() 
    n_runs = args.n_runs
    decoder = args.decoder    
    output_file = args.output_file    

    get_initial_splits(n_runs, 
                       decoder, 
                       output_file, 
                       weight_A=None, 
                       weight_B=None)


if __name__ == '__main__':
    run_threshold()
