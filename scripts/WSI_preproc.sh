#!/bin/bash

#SBATCH -t 24:00:00 
#SBATCH -c 8
#SBATCH --mem=250G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=reardons@ufl.edu

cd 
conda activate ./conda
cd CLAM

DATA_DIRECTORY=
CLAM_PATCHES_DIRECTORY=

python create_patches_fp.py --source $DATA_DIRECTORY --save_dir $CLAM_PATCHES_DIRECTORY --patch_size 224 --patch --seg --no_auto_skip \
    --preset tcga-blca.csv