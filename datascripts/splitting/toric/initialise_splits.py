import numpy as np
from qecsim.models.generic import BitFlipErrorModel
from qecsim.models.color import Color488Code
from pymatching import Matching
from qecsim import decodertools as dt
from qecsim import unifiedDecodertools as udt
import pandas as pd
import argparse


def get_initial_splits(n_runs, output_file):

    L_list = [6, 10, 14]
    p_list = [0.05] 
    results = []
    
    # run unified decoder
    for L in L_list:

        # run restricted decoder 

        my_code = Color488Code(L)
        H_full = my_code.stabilizers(restricted=None)[:, my_code.n_k_d[0]:]
        H_green = dt.get_check_matrix(my_code, 'green')
        m = Matching(H_green)

        for p in p_list:

            n_fails = 0

            for _ in range(n_runs):
                error = udt.BadEntropyErrorModel(my_code, p)
                n_fails += dt.logical_failures(my_code, error, H_full, m, 'green')

            results.append({'L':L, 'p':p, 'n_runs':n_runs, 'n_fails':n_fails, 'decoder':'restricted'})

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False) 

def run_threshold():
    parser = argparse.ArgumentParser(description='Calculate unified threshold')
    parser.add_argument('--n_runs', '-n', type=int, default=1000,
                        help='Set number of runs per script') 
    parser.add_argument('--output_file', '-o', type=str,
                    help='output file')
    args = parser.parse_args() 
    n_runs = args.n_runs
    output_file = args.output_file    

    get_initial_splits(n_runs, output_file)

if __name__ == '__main__':
    run_threshold()
