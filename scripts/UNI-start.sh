#!/bin/bash

#SBATCH -t 48:00:00
#SBATCH -c 12
#SBATCH --mem=500G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=reardons@ufl.edu
#SBATCH --partition=gpu
#SBATCH --gpus=a100:3

export UNI_CKPT_PATH=pytorch_model.bin
export LD_LIBRARY_PATH=/conda/lib:$LD_LIBRARY_PATH
DIR_TO_COORDS=
DATA_DIRECTORY=
CSV_FILE_NAME=
FEATURES_DIRECTORY=
cd CLAM

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

gpuList=$(echo $CUDA_VISIBLE_DEVICES | sed -e 's/,/ /g')
N=0
devList=""
for gpu in $gpuList
do
    devList="$devList $N"
    N=$(($N + 1))
done
devList=$(echo $devList | sed -e 's/ /,/g')
echo "devList = $devList"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python extract_features_fp.py --data_h5_dir ${DIR_TO_COORDS} --data_slide_dir ${DATA_DIRECTORY} --csv_path ${CSV_FILE_NAME} --feat_dir ${FEATURES_DIRECTORY} --batch_size 512 --slide_ext .svs --model_name uni_v1