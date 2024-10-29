from unimatch.codes.color import Color488Code
from unimatch.splitting.splittingtools import *
import argparse
import pandas as pd

# Dictionary of errors 

err_dict = {6 : {'ler_five' : 0.030655, 'init_error' : [0, 7, 4]},
     8 : {'ler_five' : 0.017266, 'init_error' : [0, 9, 2, 6]},
     10 : {'ler_five' : 0.009844, 'init_error' : [0, 11, 2, 3, 8]},
     12 : {'ler_five' : 0.0055666, 'init_error' : [0, 13, 2, 3, 4, 5]},
     14 : {'ler_five' : 0.0035, 'init_error' : [0, 15, 2, 3, 4, 5, 12]},
     16 : {'ler_five' : 0.0018, 'init_error' : [0, 17, 2, 3, 4, 5, 6, 7]},
     18 : {'ler_five' : 0.0013, 'init_error' : [0, 19, 2, 3, 4, 5, 6, 7, 16]},
     }


def run_splitting(size, n_samples, output_file):
    """
    Runs splitting sampling for the restricted decoder on code of size L.
    """
    results = []
        
    my_code = Color488Code(size)
    per = physical_error_splits(0.05, 10, my_code) # start at p = 0.05
    data_restricted = generate_splitting_data_restricted(size, per, err_dict[size]['ler_five'], err_dict[size]['init_error'], 1, n_samples) 

    for i_p, p in enumerate(per):
        results.append({'L':size, 'p':p, 'n_samples':n_samples, 'logical_p':data_restricted[i_p], 'decoder':'restricted'})
    
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