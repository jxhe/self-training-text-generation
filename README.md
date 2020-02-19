# Self-Training for Neural Sequence Generation

This repo is a PyTorch implementation of noisy self-training algorithms from the following [paper](https://arxiv.org/abs/1909.13788):

```bibtex
@inproceedings{He2020Revisiting,
title={Revisiting Self-Training for Neural Sequence Generation},
author={Junxian He and Jiatao Gu and Jiajun Shen and Marc'Aurelio Ranzato},
booktitle={Proceedings of ICLR},
year={2020},
url={https://openreview.net/forum?id=SJgdnAVKDH}
}
```

**Note: this repo is under construction, will be ready soon**



## Requirement

[fairseq](https://github.com/pytorch/fairseq) (Please see the fairseq repo for other requirements on PyTorch and Python versions)

fairseq can be installed with:

```shell
pip install fairseq
```



## Data

Download the WMT'14 En-De dataset:

```shell
# Dowload and prepare the data
cd examples/self_training/
bash ../translation/prepare-wmt14en2de.sh --icml17
```

Preprocess data:

```shell
TEXT=wmt14_en_de
fairseq-preprocess --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir wmt14_en_de_bin --thresholdtgt 0 --thresholdsrc 0 \
    --joined-dictionary --workers 16
```

Then we mimic a semi-supervised setting where 100K training samples are randomly selected as parallel corpus and the remaining English training samples are treated as unannotated monolingual corpus:

```shell
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
```

Preprocess WMT100K:

```shell
 TEXT=wmt14_en_de_bin                                                                                                                                                
 fairseq-preprocess --source-lang 100ken --target-lang 100kde \                                                                                                      
        --trainpref wmt14_en_de/train --srcdict $TEXT/dict.en.txt \                                                                                                   
        --tgtdict $TEXT/dict.de.txt \                                                                                                                                 
        --destdir $TEXT --workers 16 
```

```shell
bash link.sh
```



Train the translation model:

```shell
CUDA_VISIBLE_DEVICES=xx bash train.sh [src] [tgt]
```






