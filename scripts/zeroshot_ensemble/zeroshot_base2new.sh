#!/bin/bash

# custom config
DATA=$1
DATASET=$2
MODEL=$3
METHOD=Zeroshot
TRAINER=Zeroshot${MODEL}2

CFG=vit_b16_base2new
SUB_base=base
SUB_novel=new

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${METHOD}/${CFG}.yaml \
--output-dir output/base2new/train_base/${DATASET}/${TRAINER} \
--eval-only \
DATASET.SUBSAMPLE_CLASSES ${SUB_base}

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${METHOD}/${CFG}.yaml \
--output-dir output/base2new/test_new/${DATASET}/${TRAINER} \
--eval-only \
DATASET.SUBSAMPLE_CLASSES ${SUB_novel}