#!/bin/bash
#
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --time=10:00
#SBATCH --mem=5G


python /scratch/assignment4/test_scripts/triangle_counting_tester.pyc --execPath=/home/$USER/sfuhome/CMPT431/A4/triangle_counting_parallel
