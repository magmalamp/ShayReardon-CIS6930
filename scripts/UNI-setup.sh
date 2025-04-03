#!/bin/bash

#SBATCH -t 24:00:00 
#SBATCH -c 8
#SBATCH --mem=250G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=reardons@ufl.edu

python analysis/UNI-setup.py