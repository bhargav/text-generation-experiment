import random


# Synthetic Dataset based on:
# https://github.com/ofirnachum/sequence_gan/blob/master/simple_demo.py
class SyntheticDataset(object):
    def __init__(self, num_emb=4, seq_length=15, start_token=0):
        self.num_emb = num_emb
        self.seq_length = seq_length
        self.start_token = start_token

    def verify_sequence(self, seq):
        downhill = True
        prev = self.num_emb
        for tok in seq:
            if tok == self.start_token:
                return False
            if downhill:
                if tok > prev:
                    downhill = False
            elif tok < prev:
                return False
            prev = tok
        return True

    def get_random_sequence(self):
        """Returns random valley sequence."""
        tokens = set(range(self.num_emb))
        tokens.discard(self.start_token)
        tokens = list(tokens)

        pivot = int(random.random() * self.seq_length)
        left_of_pivot = []
        right_of_pivot = []
        for i in range(self.seq_length):
            tok = random.choice(tokens)
            if i <= pivot:
                left_of_pivot.append(tok)
            else:
                right_of_pivot.append(tok)

        left_of_pivot.sort(reverse=True)
        right_of_pivot.sort(reverse=False)

        return left_of_pivot + right_of_pivot


if __name__ == "__main__":
    data = SyntheticDataset()
    for _ in range(1000):
        sequence = data.get_random_sequence()
        print(sequence)
        assert data.verify_sequence(sequence)
