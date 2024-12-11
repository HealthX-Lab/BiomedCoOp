CFG=vit_b16
DATA=$1
DATASET=$2
MODEL=$3
METHOD=Zeroshot
TRAINER=Zeroshot${MODEL}

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${METHOD}/${CFG}.yaml \
--output-dir output/${DATASET}/${TRAINER}/${CFG} \
--eval-only