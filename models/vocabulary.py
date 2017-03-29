"""Utilities for parsing text sentences and building a vocabulary"""

import collections
import dill as pickle

UNK_WORD = '<unk_word>'  # Unknown word
EOS = '<eos>'  # End-of-Sentence Token


class vocabulary(object):

    def __init__(self, data=[]):
        data.append(UNK_WORD)
        data.append(EOS)

        self._counter = counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))

        self.vocab = word_to_id

    def save_to_disk(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump({"counter": self._counter, "vocab": self.vocab}, file)

    @staticmethod
    def load_vocabulary(file_name):
        with open(file_name, 'r') as file:
            items = pickle.load(file)

        vocab_instance = vocabulary()
        vocab_instance._counter = items["counter"]
        vocab_instance.vocab = items["vocab"]

        return vocab_instance


if __name__ == "__main__":
    test_data = ["a", "aa", "ba", "aa", "c"]

    v = vocabulary(test_data)
    print(v.vocab)
