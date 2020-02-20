import torch
import numpy as np


class WordNoising(object):
    """Generate a noisy version of a sentence, without changing words themselves."""
    def __init__(self, dictionary, bpe_type="subword",
        bos="<bos>", eos="<eos>", unk="<unk>", pad="<pad>"):
        self.dictionary = dictionary
        self.bpe_end = None
        self.bpe_start = None
        self.bos = bos
        self.eos = eos
        self.unk = unk
        self.pad = pad

        self.bpe_type = bpe_type

        if bpe_type == "subword":
            self.bpe_end = np.array([
                not self.dictionary.itos[i].endswith("@@")
                for i in range(len(self.dictionary.itos))
            ])
        elif bpe_type == "sentencepiece":
            self.bpe_start = np.array([
                self.dictionary.itos[i].startswith("\u2581") \
                for i in range(len(self.dictionary.itos))
            ])
        elif bpe_type == "wordpiece":
            self.bpe_start = np.array([
                not self.dictionary.itos[i].startswith("##")
                for i in range(len(self.dictionary.itos))
            ])            
        else:
            raise ValueError("the bpe type is not supported")

        self.get_word_idx = self._get_bpe_word_idx

    def noising(self, x, lengths, noising_prob=0.0):
        raise NotImplementedError()

    def _get_bpe_word_idx(self, x):
        """
        Given a list of BPE tokens, for every index in the tokens list,
        return the index of the word grouping that it belongs to.
        For example, for input x corresponding to ["how", "are", "y@@", "ou"],
        return [[0], [1], [2], [2]].
        """
        # x: (T x B)
        if self.bpe_type == "subword":
            bpe_end = self.bpe_end[x]

            if (x.size(0) == 1 and x.size(1) == 1):
                # Special case when we only have one word in x. If x = [[N]],
                # bpe_end is a scalar (bool) instead of a 2-dim array of bools,
                # which makes the sum operation below fail.
                return np.array([[0]])

            # do a reduce front sum to generate word ids
            word_idx = bpe_end[::-1].cumsum(0)[::-1]
            word_idx = word_idx.max(0)[None, :] - word_idx
            return word_idx
        else:
            bpe_start = self.bpe_start[x]
            if (x.size(0) == 1 and x.size(1) == 1):
                return np.array([[0]])
            cumsum = bpe_start.cumsum(0)
            word_idx = cumsum - cumsum.min(0, keepdims=True)
            return word_idx


class WordDropout(WordNoising):
    """Randomly drop input words. If not passing blank_idx (default is None),
    then dropped words will be removed. Otherwise, it will be replaced by the
    blank_idx."""

    def __init__(self, dictionary, default_dropout_prob=0.1, bpe_type="subword",
        bos="<bos>", eos="<eos>", unk="<unk>", pad="<pad>"):
        super().__init__(dictionary, bpe_type,
            bos, eos, unk ,pad)
        self.default_dropout_prob = default_dropout_prob

    def noising(self, x, lengths, mask_bpe=False, dropout_prob=None, blank_idx=None):
        if dropout_prob is None:
            dropout_prob = self.default_dropout_prob
        # x: (T x B), lengths: B
        if dropout_prob == 0:
            return x, lengths

        assert 0 < dropout_prob < 1

        # be sure to drop entire words
        word_idx = self.get_word_idx(x)
        sentences = []
        modified_lengths = []
        for i in range(lengths.size(0)):
            # Since dropout probabilities need to apply over non-pad tokens,
            # it is not trivial to generate the keep mask without consider
            # input lengths; otherwise, this could be done outside the loop

            # We want to drop whole words based on word_idx grouping
            num_words = max(word_idx[:, i]) + 1

            # ith example: [x0, x1, ..., eos, pad, ..., pad]
            # We should only generate keep probs for non-EOS tokens. Thus if the
            # input sentence ends in EOS, the last word idx is not included in
            # the dropout mask generation and we append True to always keep EOS.
            # Otherwise, just generate the dropout mask for all word idx
            # positions.
            has_eos = x[lengths[i] - 1, i] == self.dictionary.stoi[self.eos]
            if has_eos:  # has eos?
                num_words = word_idx[lengths[i]-1, i] + 1
                keep = np.random.rand(num_words - 1) >= dropout_prob
                keep = np.append(keep, [True])  # keep EOS symbol
            else:
                keep = np.random.rand(num_words) >= dropout_prob

            words = x[:lengths[i], i].tolist()

            # TODO: speed up the following loop
            # drop words from the input according to keep
            if mask_bpe:
                new_s = [
                    w if keep[word_idx[j, i]] else blank_idx
                    for j, w in enumerate(words)
                ]
                new_s = [w for w in new_s if w is not None]
            else:
                new_s = []
                last = -1
                for j, w in enumerate(words):
                    if keep[word_idx[j, i]]:
                        new_s.append(w)
                    elif blank_idx is not None:
                        if word_idx[j, i] != last:
                            new_s.append(blank_idx)

                    last = word_idx[j, i]

            # we need to have at least one word in the sentence (more than the
            # start / end sentence symbols)
            if len(new_s) <= 1:
                # insert at beginning in case the only token left is EOS
                # EOS should be at end of list.
                new_s.insert(0, words[np.random.randint(0, len(words))])
            assert len(new_s) >= 1 and (
                not has_eos  # Either don't have EOS at end or last token is EOS
                or (len(new_s) >= 2 and new_s[-1] == self.dictionary.stoi[self.eos])
            ), "New sentence is invalid."
            sentences.append(new_s)
            modified_lengths.append(len(new_s))
        # re-construct input
        modified_lengths = torch.LongTensor(modified_lengths)
        modified_x = torch.LongTensor(
            modified_lengths.max(),
            modified_lengths.size(0)
        ).fill_(self.dictionary.stoi[self.pad])
        for i in range(modified_lengths.size(0)):
            modified_x[:modified_lengths[i], i].copy_(torch.LongTensor(sentences[i]))

        return modified_x, modified_lengths


class WordShuffle(WordNoising):
    """Shuffle words by no more than k positions."""

    def __init__(self, dictionary, default_max_shuffle_distance=3, bpe_type="subword",
        bos="<bos>", eos="<eos>", unk="<unk>", pad="<pad>"):
        super().__init__(dictionary, bpe_type,
            bos, eos, unk ,pad)
        self.default_max_shuffle_distance = 3

    def noising(self, x, lengths, max_shuffle_distance=None):
        if max_shuffle_distance is None:
            max_shuffle_distance = self.default_max_shuffle_distance
        # x: (T x B), lengths: B
        if max_shuffle_distance == 0:
            return x, lengths

        # max_shuffle_distance < 1 will return the same sequence
        assert max_shuffle_distance > 1

        # define noise word scores
        noise = np.random.uniform(
            0,
            max_shuffle_distance,
            size=(x.size(0), x.size(1)),
        )
        noise[0] = -1  # do not move start sentence symbol
        # be sure to shuffle entire words
        word_idx = self.get_word_idx(x)
        x2 = x.clone()
        for i in range(lengths.size(0)):
            length_no_eos = lengths[i]
            if x[lengths[i] - 1, i] == self.dictionary.stoi[self.eos]:
                length_no_eos = lengths[i] - 1
            # generate a random permutation
            scores = word_idx[:length_no_eos, i] + noise[word_idx[:length_no_eos, i], i]
            # ensure no reordering inside a word
            scores += 1e-6 * np.arange(length_no_eos)
            permutation = scores.argsort()
            # shuffle words
            x2[:length_no_eos, i].copy_(
                x2[:length_no_eos, i][torch.from_numpy(permutation)]
            )
        return x2, lengths


class UnsupervisedMTNoising(WordNoising):
    """
    Implements the default configuration for noising in UnsupervisedMT
    (github.com/facebookresearch/UnsupervisedMT)
    """
    def __init__(
        self,
        dictionary,
        max_word_shuffle_distance,
        word_dropout_prob,
        word_blanking_prob,
        bpe_type="subword",
        mask_bpe=False,
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        pad="<pad>"
    ):
        super().__init__(dictionary)
        self.max_word_shuffle_distance = max_word_shuffle_distance
        self.word_dropout_prob = word_dropout_prob
        self.word_blanking_prob = word_blanking_prob

        self.unk = unk
        self.mask_bpe = mask_bpe

        self.word_dropout = WordDropout(
            dictionary=dictionary,
            bpe_type=bpe_type,
            bos=bos,
            eos=eos,
            unk=unk,
            pad=pad
        )
        self.word_shuffle = WordShuffle(
            dictionary=dictionary,
            bpe_type=bpe_type,
            bos=bos,
            eos=eos,
            unk=unk,
            pad=pad
        )

    def noising(self, x, lengths):
        # 1. Word Shuffle
        noisy_src_tokens, noisy_src_lengths = self.word_shuffle.noising(
            x=x,
            lengths=lengths,
            max_shuffle_distance=self.max_word_shuffle_distance,
        )
        # 2. Word Dropout
        noisy_src_tokens, noisy_src_lengths = self.word_dropout.noising(
            x=noisy_src_tokens,
            lengths=noisy_src_lengths,
            mask_bpe=self.mask_bpe,
            dropout_prob=self.word_dropout_prob,
        )
        # 3. Word Blanking
        noisy_src_tokens, noisy_src_lengths = self.word_dropout.noising(
            x=noisy_src_tokens,
            lengths=noisy_src_lengths,
            mask_bpe=self.mask_bpe,
            dropout_prob=self.word_blanking_prob,
            blank_idx=self.dictionary.stoi[self.unk],
        )

        return noisy_src_tokens, noisy_src_lengths
