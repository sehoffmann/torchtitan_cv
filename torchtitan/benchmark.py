import torch

class ConstantDataloader:

    def __init__(self, batch_size, sequence_length, vocab_size, device):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self._data = torch.randint(vocab_size, size=(batch_size, sequence_length), device=device)
        self._labels = torch.randint(vocab_size, size=(batch_size, sequence_length), device=device)

    def __iter__(self):
        while True:
            yield self._data, self._labels