#! /bin/bash

SAVE_ROOT=checkpoint

mkdir -p ${SAVE_ROOT}

model=transformer_wmt_en_de
dropout=0.3
src=$1
tgt=$2

train_steps=${3:-300000}


SAVE=${SAVE_ROOT}/${model}.drop${dropout}.${src}-${tgt}.nsteps${train_steps}
# TENSORBOARD=${SAVE}/tensorboard
mkdir -p ${SAVE}

# ---- supervised training ---- #

fairseq-train wmt14_en_de_bin \
     -a ${model} --optimizer adam --lr 0.0005 -s ${src} -t ${tgt} \
     --dropout ${dropout} --max-tokens 4096 \
     --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-update ${train_steps} \
     --warmup-updates 4000 --warmup-init-lr '1e-07' \
     --adam-betas '(0.9, 0.98)' --save-dir ${SAVE} \
     --task translation \
     --log-format simple --log-interval 30 \
     --share-all-embeddings \
     --save-interval-updates 100 \
     --update-freq 1 --fp16 \
     --validate-interval 1000 --keep-interval-updates 10 --save-interval 1000\
    | tee -a ${SAVE}/stdout.log
wait $!

# ---- check performance ---- #

fairseq-generate wmt14_en_de_bin --source-lang en --target-lang de \
    --path ${SAVE}/checkpoint_last.pt --beam 5 --batch-size 128 --remove-bpe \
    > ${SAVE}/gen_last.out
wait $!

fairseq-generate wmt14_en_de_bin --source-lang en --target-lang de \
    --path ${SAVE}/checkpoint_best.pt --beam 5 --batch-size 128 --remove-bpe \
    > ${SAVE}/gen_best.out
wait $!

