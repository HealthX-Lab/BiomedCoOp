#!/bin/bash

# custom config
DATA=$1
DATASET=$2
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
MODEL=$4

METHOD=TIPAdapter_${MODEL}


python main.py \
 --base_config configs/trainers/TiP_Adapter/few_shot.yaml \
 --dataset_config configs/datasets/${DATASET}.yaml \
 --opt root_path ${DATA} \
   output_dir output/${DATASET}/shots_${SHOTS}/${METHOD}/ \
   shots ${SHOTS} \
   method ${METHOD} \
   clip_model ${MODEL} 


