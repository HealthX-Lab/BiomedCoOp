# custom config
DATA=$1
DATASET=$2
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
MODEL=BiomedCLIP
NCTX=4
CSC=False
CTP=end
LOADEP=100

METHOD=BiomedCoOp
TRAINER=BiomedCoOp_${MODEL}

for SEED in 1 2 3
do
        MODEL_DIR=few_shot/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
        if [ -d "$MODEL_DIR" ]; then
            echo "The checkpoint exists at ${MODEL_DIR} (skipping to evaluation)"
        else
            python download_ckpts.py \
            --task few_shot \
            --dataset ${DATASET} \
            --shots ${SHOTS} \
            --trainer ${TRAINER}
            echo "Downloaded the checkpoint for ${MODEL_DIR}"
        fi
        DIR=output_eval/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
           python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${METHOD}/few_shot/${DATASET}.yaml  \
            --model-dir ${MODEL_DIR} \
            --load-epoch ${LOADEP} \
            --output-dir ${DIR} \
            --eval-only \
            TRAINER.BIOMEDCOOP.N_CTX ${NCTX} \
            TRAINER.BIOMEDCOOP.CSC ${CSC} \
            TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION ${CTP}  
        fi
done