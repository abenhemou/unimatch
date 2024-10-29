from unimatch.codes.color import Color488Code
from unimatch.splitting.splittingtools import *
import argparse
import pandas as pd

# Dictionary of initial errors 

err_dict = {6 : {'ler_five' : 0.0297, 'init_error' : [0, 7, 4]},
     8 : {'ler_five' : 0.0176, 'init_error' : [84, 93, 97, 98]},
     10 : {'ler_five' : 0.009, 'init_error' : [0, 11, 3, 8, 14]},
     12 : {'ler_five' : 0.0031, 'init_error' : [0, 13, 3, 6, 10, 16]},
     14 : {'ler_five' : 0.0035, 'init_error' : [0, 15, 3, 4, 5, 7, 12]},
     }

def run_splitting(size, n_samples, output_file):
    """
    Runs splitting sampling for the unified decoder on code of size L.
    """
    results = []
    
    my_code = Color488Code(size)
    per = physical_error_splits(0.05, 20, my_code) # start at p = 0.05
    data_unified = generate_splitting_data_unified(size, per, err_dict[size]['ler_five'], err_dict[size]['init_error'], 1, n_samples) 

    for i_p, p in enumerate(per):
        results.append({'L':size, 'p':p, 'logical_p':data_unified[i_p], 'decoder':'unified'})
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)


def run_splits():

    parser = argparse.ArgumentParser(description='Calculate low p error rates using splitting method')

    parser.add_argument('--size', '-L', type=int, default=4,
                        help='Lattice size.') 
    parser.add_argument('--n_samples', '-n', type=int, default=1000,
                        help='Set number of Metropolis samples per script')
    parser.add_argument('--output_file', '-o', type=str,
                    help='output file')
    args = parser.parse_args()
    size = args.size
    n_samples = args.n_samples
    output_file = args.output_file

    run_splitting(size, n_samples, output_file)


if __name__ == '__main__':
    run_splits()