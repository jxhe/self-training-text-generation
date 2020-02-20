#! /bin/bash

TEXT=wmt14_en_de_bin
src=${1}
tgt=${2}
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
       --trainpref wmt14_en_de/train --srcdict $TEXT/dict.en.txt \
       --tgtdict $TEXT/dict.de.txt \
       --destdir $TEXT --workers 16

cd ${TEXT}
ln -s valid.en-de.en.idx valid.${src}-${tgt}.${src}.idx
ln -s valid.en-de.en.bin valid.${src}-${tgt}.${src}.bin
ln -s valid.en-de.de.idx valid.${src}-${tgt}.${tgt}.idx
ln -s valid.en-de.de.bin valid.${src}-${tgt}.${tgt}.bin

cd ..
