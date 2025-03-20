# custom config
DATA=$1
DATASET=$2
SHOTS=16
MODEL=BiomedCLIP
NCTX=4
CSC=False
CTP=end
LOADEP=50

METHOD=BiomedCoOp
TRAINER=BiomedCoOp_${MODEL}
SUB_base=base
SUB_novel=new

for SEED in 1 2 3
do
    MODEL_DIR=base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$MODEL_DIR" ]; then
        echo "The checkpoint exists at ${MODEL_DIR} (skipping to evaluation)"
    else
        python download_ckpts.py \
        --task base2new \
        --dataset ${DATASET} \
        --trainer ${TRAINER}
        echo "Downloaded the checkpoint for ${MODEL_DIR}"
    fi
    for SUB_TASK in "base" "new"
        do
        DIR=output_eval/base2new/test_${SUB_TASK}/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
            python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${METHOD}/base_to_novel/${DATASET}.yaml  \
                --model-dir ${MODEL_DIR} \
                --load-epoch ${LOADEP} \
                --output-dir ${DIR} \
                --eval-only \
                DATASET.SUBSAMPLE_CLASSES ${SUB_TASK}
        fi
        done
done