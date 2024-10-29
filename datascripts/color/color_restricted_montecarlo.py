import numpy as np
from unimatch.codes import BitFlipErrorModel
from unimatch.codes.color import Color488Code
from pymatching import Matching
from unimatch.decoders import decodertools as dt
import pandas as pd
import argparse

my_error_model = BitFlipErrorModel()

def run_restricted_decoder(size, n_runs, output_file):

    L = size 
    p_list = np.linspace(0.10, 0.103, 10)
    results = []

    my_code = Color488Code(L)
    H_full = my_code.stabilizers(restricted=None)[:, my_code.n_k_d[0]:]
    H_green = dt.get_check_matrix(my_code, 'green')
    m = Matching(H_green) 

    for p in p_list:   

        n_fails = 0

        for _ in range(n_runs):
            error = my_error_model.generate(my_code, p)[0:my_code.n_k_d[0]]
            n_fails += dt.logical_failures(my_code, error, H_full, m, 'green')

        results.append({'L':L, 'p':p, 'n_runs':n_runs, 'n_fails':n_fails, 'decoder':'restricted'}) # 'ler': n_runs/n_fails

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

def run_threshold():
    parser = argparse.ArgumentParser(description='Calculate restricted threshold')

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




