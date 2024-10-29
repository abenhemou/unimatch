#!/bin/bash -l
# Request wallclock time (format hours:minutes:seconds).
#$ -l h_rt=03:00:00
#$ -l mem=15G
#$ -l tmpfs=10G
#$ -N L20thrColor
#$ -pe smp 5

#$ -M asmae.benhemou.19@ucl.ac.uk
#$ -wd /home/ucapenh/Scratch/output

module purge
module load parallel/20181122

source ~/.bashrc

# Load python env
conda activate ColorCodes 
# Variables
n_total=10000 # total number of runs
n_runs=100 # number of runs per script
n_scripts=$n_total/$n_runs  
n_cores=5
n_nodes=$n_scripts/$n_cores
i_task=$(( SGE_TASK_ID-1 ))

#$ -t 1-20

L=12 # Set code distance 
script_dir="." # Specify directory containing scripts 
output_dir="." # Specify output directory for data 

function get_parameters {
    for ((i=0; i<n_cores; i++)); do
        if [ $(( i+i_task*n_cores )) -le $n_total ]; then 
            echo $(( i+i_task*n_cores )); 
        fi
    done 
}

# Bash command to parallelize
bash_command="python $script_dir/color_restricted_montecarlo.py -L $L -n $n_runs -o $output_dir/color_restricted_L_$((L))_$(date +"%Y_%m_%d_%I_%M_%p")_{1}.csv" 

# Run in parallel | parallelise takes the list of the entries that replace {1} in bash_command 
get_parameters | parallel "$bash_command"

# Print out the date when done because why not
date

