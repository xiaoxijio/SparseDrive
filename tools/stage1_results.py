"""Read locally produced Stage 1 prediction pickles without allocating CUDA memory."""
import io
import pickle
import torch


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda value: torch.load(io.BytesIO(value), map_location='cpu')
        return super().find_class(module, name)


def load_results(path):
    with open(path, 'rb') as stream:
        return CPUUnpickler(stream).load()
