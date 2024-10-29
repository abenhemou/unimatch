import numpy as np
from unimatch.codes.color import Color488Code
from pymatching import Matching
from unimatch.decoders import unifiedColor488Decoder as udt
from unimatch.codes import BitFlipErrorModel
import pandas as pd
import argparse

my_error_model = BitFlipErrorModel()

# def weight_tailored_at_boundaries(my_code, ids):
#     '''
#     Return weight of edge in lattice L depending on its position.
#     '''
#     rg_sides = list(set(my_code.boundaries[0] + my_code.boundaries[1]) - set(my_code.corners))
#     rb_sides = list(set(my_code.boundaries[2] + my_code.boundaries[3]) - set(my_code.corners))
    
#     if any(id in my_code.corners for id in ids):
#         weight = 2
#     elif any(id in rb_sides for id in ids):
#         weight = 2
#     elif any(id in rg_sides for id in ids):
#         weight = 1
#     else:
#         weight = 1
#     return weight 

def run_unified_decoder(size, weight, n_runs, output_file):
    
    p_list = np.linspace(0.096, 0.108, 5)

    results = []
    L = size

    my_code = Color488Code(L)
    H_full = my_code.stabilizers(restricted=None)[:, my_code.n_k_d[0]:]
    H_green = udt.get_check_matrix(my_code, 'green')
    H_blue = udt.get_check_matrix(my_code, 'blue')
    H_red = udt.get_check_matrix(my_code, 'red')
    check_matrix = udt.expand_check_matrix(H_red, H_blue, H_green, my_code)
    H_unified = udt.unified_check_matrix(check_matrix, L, my_code)
    H_graph, _ = udt.restrict_unified_check_matrix(H_unified)
    m = Matching(H_graph)
    weights = [udt.test_weights_config_c(my_code, udt.get_fault_ids(H_unified, (e[0],e[1])), [weight, 1]) for e in m.edges()]
    
    # Or tailor weights using function above 
    # weights = [weight_tailored_at_boundaries(my_code, udt.get_fault_ids(H_unified, (e[0],e[1]))) for e in m.edges()]
    m_with_weights = Matching()
    m_with_weights.load_from_check_matrix(H_graph, spacelike_weights=weights)
    logical_crossings = [tuple(np.where(H_unified[:,i])[0]) for i in my_code.boundaries[3]]

    for p in p_list:   

        n_fails = 0

        for _ in range(n_runs):
            error = my_error_model.generate(my_code, p)[0:my_code.n_k_d[0]]
            syndrome = H_full@error % 2 
            n_fails += udt.logical_failures(my_code, error, syndrome, logical_crossings, m_with_weights)

        results.append({'L':L, 'p':p, 'n_runs':n_runs, 'n_fails':n_fails, 'weight':weight, 'decoder':'unified'})

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)


def run_threshold():
    parser = argparse.ArgumentParser(description='Calculate unified threshold')

    parser.add_argument('--size', '-L', type=int,
                    help='Lattice size')
    parser.add_argument('--weight', '-A', type=float,
                    help='Weight of the blue and green lattices')
    parser.add_argument('--n_runs', '-n', type=int, default=1000,
                        help='Set number of runs per script') 
    parser.add_argument('--output_file', '-o', type=str,
                    help='output file')

    args = parser.parse_args()
    size = args.size
    weight = args.weight
    n_runs = args.n_runs
    output_file = args.output_file 

    run_unified_decoder(size, weight, n_runs, output_file)

if __name__ == '__main__':
    run_threshold()