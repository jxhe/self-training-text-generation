#! /bin/bash

model_path=${1}
suffix=${2:-mono_de_iter1}

cat wmt14_en_de/train.mono_en | \
    fairseq-interactive wmt14_en_de_bin \
    --path ${model_path} \
    --beam 5 --fp16 --batch-size 100 \
    --buffer-size 100 > wmt14_en_de/train.mono_en.gen

grep ^H wmt14_en_de/train.mono_en.gen | cut -f3- > wmt14_en_de/train.${suffix}

rm wmt14_en_de/train.mono_en.gen
