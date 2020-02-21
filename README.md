# Self-Training for Neural Sequence Generation

This repo includes instructions for running noisy self-training algorithms from the following [paper](https://arxiv.org/abs/1909.13788):

```
Revisiting Self-Training for Neural Sequence Generation
Junxian He*, Jiatao Gu*, Jiajun Shen, Marc'Aurelio Ranzato
ICLR 2020
```



## Requirement

- [fairseq](https://github.com/pytorch/fairseq) (please see the fairseq repo for other requirements on Python and PyTorch versions)



fairseq can be installed with:

```shell
pip install fairseq
```



## Data

Download and preprocess the WMT'14 En-De dataset:

```shell
# Download and prepare the data
wget https://raw.githubusercontent.com/pytorch/fairseq/master/examples/translation/prepare-wmt14en2de.sh
bash prepare-wmt14en2de.sh --icml17

TEXT=wmt14_en_de
fairseq-preprocess --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir wmt14_en_de_bin --thresholdtgt 0 --thresholdsrc 0 \
    --joined-dictionary --workers 16
```

Then we mimic a semi-supervised setting where 100K training samples are randomly selected as parallel corpus and the remaining English training samples are treated as unannotated monolingual corpus:

```shell
bash extract_wmt100k.sh
```



Preprocess WMT100K:

```shell
bash preprocess.sh 100ken 100kde 
```



Add noise to the monolingual corpus for later usage:

```shell
TEXT=wmt14_en_de
python paraphrase/paraphrase.py \
  --paraphraze-fn noise_bpe \
  --word-dropout 0.2 \
  --word-blank 0.2 \
  --word-shuffle 3 \
  --data-file ${TEXT}/train.mono_en \
  --output ${TEXT}/train.mono_en_noise \
  --bpe-type subword
```



## Train the base supervised model

Train the translation model with 30K updates:

```shell
bash supervised_train.sh 100ken 100kde 30000
```



## Self-training as pseudo-training + fine-tuning

Translate the monolingual data to `train.[suffix]` to form a pseudo parallel dataset:

```shell
bash translate.sh [model_path] [suffix]  
```



Suppose the pseduo target language `suffix` is `mono_de_iter1` (by default), preprocess:

```shell
bash preprocess.sh mono_en_noise mono_de_iter1
```



Pseudo-training + fine-tuning: 

```shell
bash self_train.sh mono_en_noise mono_de_iter1 
```

The above command trains the model on the pseduo parallel corpus formed by `train.mono_en_noise` and `train.mono_de_iter1` and then fine-tune it on real parallel data.



This self-training process can be repeated for multiple iterations to improve performance.



## Reference

```bibtex
@inproceedings{He2020Revisiting,
title={Revisiting Self-Training for Neural Sequence Generation},
author={Junxian He and Jiatao Gu and Jiajun Shen and Marc'Aurelio Ranzato},
booktitle={Proceedings of ICLR},
year={2020},
url={https://openreview.net/forum?id=SJgdnAVKDH}
}
```

