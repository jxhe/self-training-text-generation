import argparse
import random
import io
import numpy as np

import torch
import torchtext

from tqdm import tqdm
from torchtext.data import Field, Iterator
from torchtext.datasets import LanguageModelingDataset

from noise import NoiseLayer
from noise_bpe import UnsupervisedMTNoising

class SentenceModelingDataset(torchtext.data.Dataset):
    """Defines a dataset for language modeling."""

    def __init__(self, path, text_field, newline_eos=True,
                 encoding='utf-8', **kwargs):
        """Create a LanguageModelingDataset given a path and a field.
        Arguments:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            newline_eos: Whether to add an <eos> token for every newline in the
                data file. Default: True.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field)]
        examples = []
        with io.open(path, encoding=encoding) as f:
            for line in f:
                examples.append(torchtext.data.Example.fromlist([line], fields))

        # examples = [data.Example.fromlist([text], fields)]
        super(SentenceModelingDataset, self).__init__(
            examples, fields, **kwargs)


class ReversibleField(Field):
    def __init__(self, **kwargs):
        super(ReversibleField, self).__init__(**kwargs)

    def reverse(self, batch, length):
        """the batch tensor is required to have bos and eos tokens
        """
        if not self.batch_first:
            batch = batch.t()
        with torch.cuda.device_of(batch):
            batch = batch.tolist()

        # remove bos token
        batch = [[self.vocab.itos[ex[id_]] for id_ in range(1, len_)] for ex, len_ in zip(batch, length)]  # denumericalize

        return [' '.join(ex) for ex in batch]


def parse_args():
    parser = argparse.ArgumentParser(description="paraphrase")

    parser.add_argument("--data-file", type=str, help="input monolingual dataset")
    parser.add_argument("--paraphraze-fn", type=str, choices=["noise", "noise_bpe"],
        default="noise_bpe", help="paraphrase function")
    parser.add_argument("--bpe-type", type=str, choices=["subword", "sentencepiece", "wordpiece"],
        default="subword", help="paraphrase function")
    parser.add_argument("--output", type=str, help="output file")
    parser.add_argument("--seed", type=int, default=783435, help="random seed")
    parser.add_argument("--mask-bpe", action="store_true", default=False,
        help="if false, a whole word mask is always generating one single unk token.")

    # noise
    parser.add_argument("--word-dropout", type=float, default=0.2, help="word dropout rate")
    parser.add_argument("--word-blank", type=float, default=0.2, help="word blank rate")
    parser.add_argument("--word-shuffle", type=float, default=3., help="word shuffle rate")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    return args

def main(args):
    if args.bpe_type == "sentencepiece":
        bos = "\u2581<bos>"
        eos = "\u2581<eos>"
        unk = "\u2581<unk>"
        pad = "\u2581<pad>"
    else:
        bos = "<bos>"
        eos = "<eos>"
        unk = "<unk>"
        pad = "<pad>"
    tokenize = lambda x: [bos] + x.split() + [eos]
    TEXT = ReversibleField(sequential=True, tokenize=tokenize, include_lengths=True,
        pad_token=pad, unk_token=unk)

    dataset = SentenceModelingDataset(path=args.data_file, text_field=TEXT)

    print("complete reading data")

    TEXT.build_vocab(dataset)
    data_iter = Iterator(dataset, batch_size=32, shuffle=False, sort=False, sort_within_batch=False)

    vocab = TEXT.vocab

    if args.paraphraze_fn == "noise":
        noise_layer = NoiseLayer(args.word_blank, args.word_dropout, args.word_shuffle,
            pad_index=vocab.stoi[pad], blank_index=vocab.stoi[unk],
            eos_index=vocab.stoi[eos])
    elif args.paraphraze_fn == "noise_bpe":
        noise_layer = UnsupervisedMTNoising(vocab, args.word_shuffle, args.word_dropout,
            args.word_blank, bpe_type=args.bpe_type, mask_bpe=args.mask_bpe,
            bos=bos, eos=eos, unk=unk, pad=pad)

    pbar = tqdm(total=len(dataset))

    with open(args.output, "w", encoding="utf-8") as fout:
        for iter_ in data_iter:
            src_token, src_len = iter_.text
            prp_token, prp_len = noise_layer.noising(src_token, src_len)

            # remove <eos> symbol in the output
            reverse_len = [len_ - 1 for len_ in prp_len]

            prp_text = TEXT.reverse(prp_token, reverse_len)

            _ = [fout.write(line + "\n") for line in prp_text]

            pbar.update(len(src_len))

if __name__ == '__main__':
    args = parse_args()
    main(args)
