#! /bin/bash

cd wmt14_en_de
# concatenate and shuffle
paste -d '|' train.en train.de | cat | shuf > train.shuffle

# split
head -100000 train.shuffle > train.100kpara
tail --lines=+100001 train.shuffle > train.mono

# extract English/German
cut -d'|' -f1 train.100kpara > train.100ken
cut -d'|' -f2 train.100kpara > train.100kde
cut -d'|' -f1 train.mono > train.mono_en

rm train.shuffle train.100kpara train.mono
cd ..
