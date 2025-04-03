#!/bin/bash

#SBATCH -t 24:00:00 
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=reardons@ufl.edu
#SBATCH --partition=gpu
#SBATCH --gpus=a100:2

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python models/transMLP.py