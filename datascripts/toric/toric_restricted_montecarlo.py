import numpy as np
from unimatch.codes.color import Color488Code
import unimatch.decoders.decodertools as dt
from unimatch.codes import BadEntropyErrorModel
from pymatching import Matching
import pandas as pd
import argparse

def run_restricted_decoder(size, n_runs, output_file):

    # p_list = np.linspace(0.12, 0.17, 20)
    # p_list =  np.linspace(0.005, 0.05, 20) 
    # p_list =  np.linspace(0.145, 0.165, 20) 
    # p_list = np.linspace(0.02, 0.05, 30) 

    p_list = np.linspace(0.145, 0.16, 25) 

    # p = [0.05]

    results = []
    L = size 

    my_code = Color488Code(L)
    H_full = my_code.stabilizers(restricted=None)[:, my_code.n_k_d[0]:]
    H_green = dt.get_check_matrix(my_code, 'green')
    m = Matching(H_green)
    
    for p in p_list:

        n_fails = 0

        for _ in range(n_runs):
            error = BadEntropyErrorModel(my_code, p)
            n_fails += dt.logical_failures(my_code, error, H_full, m, 'green')
        results.append({'L':L, 'p':p, 'n_runs':n_runs, 'n_fails':n_fails, 'decoder':'restricted'}) # 'ler': n_runs/n_fails

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)


def run_threshold():
    parser = argparse.ArgumentParser(description='Calculate threshold')
    parser.add_argument('--size', '-L', type=int,
                    help='Lattice size')
    parser.add_argument('--n_runs', '-n', type=int, default=1000,
                        help='Set number of runs per script') 
    parser.add_argument('--output_file', '-o', type=str,
                    help='output file')
    args = parser.parse_args()
    size = args.size
    n_runs = args.n_runs
    output_file = args.output_file    

    run_restricted_decoder(size, n_runs, output_file)

if __name__ == '__main__':
    run_threshold()


# python /Users/asmaesmac/Research/OpenSource/datascripts/toric/threshold_toric_restricted.py -L 4 -n 5 -o test.csv


