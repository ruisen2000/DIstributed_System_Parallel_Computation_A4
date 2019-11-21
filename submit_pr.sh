#!/bin/bash
#
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --time=10:00
#SBATCH --mem=5G


srun /home/$USER/sfuhome/CMPT431/A4/./page_rank_parallel  --strategy 0 
srun /home/$USER/sfuhome/CMPT431/A4/./page_rank_parallel  --strategy 1
#srun /home/$USER/sfuhome/CMPT431/A4/./page_rank_parallel  --strategy 2 
#srun /home/$USER/sfuhome/CMPT431/A4/./page_rank_parallel  --strategy 3

