#!/bin/bash -l
# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=02:00:00
#$ -l mem=16G
#$ -l tmpfs=10G
#$ -N InitSplit5pcR
#$ -pe smp 5
#$ -wd /home/ucapenh/Scratch/output 

module purge
module load parallel/20181122

source ~/.bashrc

# Load python env
conda activate ColorCodes

# Variables
n_total=60000 # total number of runs
n_runs=600 # number of runs per script
n_scripts=$n_total/$n_runs  
n_cores=5
n_nodes=$n_scripts/$n_cores
i_task=$(( SGE_TASK_ID-1 ))

#$ -t 1-20

script_dir="." # Set dir
output_dir="." # Set dir

function get_parameters {
    for ((i=0; i<n_cores; i++)); do
        if [ $(( i+i_task*n_cores )) -le $n_total ]; then 
            echo $(( i+i_task*n_cores )); 
        fi
    done
}

# Bash command to parallelize
bash_command="python $script_dir/initialise_splits.py -n $n_runs -o $output_dir/initial_splits_uni_5pc_20000_$(date +"%Y_%m_%d_%I_%M_%p")_{1}.csv"

# Run in parallel | parallelise takes the list of the entries that replace {1} in bash_command 
get_parameters | parallel "$bash_command"

# Print out the date when done because why not
date

# python /Users/asmaesmac/Research/OpenSource/datascripts/splitting/color/initialise_splits.py -n 5 -dc restricted -o testou.csv
