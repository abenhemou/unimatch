import pandas as pd
import glob
import os
import argparse


def merge_csvdata(files_folder, output_file, group_params):
    """
    Concatenates dataframes from a list of csv files
    Requires importing pandas as pd, glob, os
    """
    all_files = glob.glob(os.path.join(files_folder, '*.csv'))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file)
    results = concatenated_df.groupby(group_params).sum()
    results['p_log'] = results['n_fails']/results['n_runs'] # Create p_log column
    results.to_csv(output_file) #, index=False)


def merge_data():

    parser = argparse.ArgumentParser(description='Merge csv data from a chosen directory')
    parser.add_argument('--files_folder', '-f', type=str,
                    help='folder with csv files to be merged')
    parser.add_argument('--output_file', '-o', type=str,
                    help='output file')
    parser.add_argument('--group_params', '-gp', type=str, default=['L', 'p', 'decoder'], 
                    help='parameters to group dataframes by') #  'weight'
    args = parser.parse_args()
    files_folder = args.files_folder   
    output_file = args.output_file
    group_params = args.group_params


    merge_csvdata(files_folder, output_file, group_params)

if __name__ == '__main__':
    merge_data()
