import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapTokenBlockDataset(Dataset):
    """Random-access fixed-length blocks backed by a compact uint16 token file."""

    def __init__(self, token_path: str, context_length: int):
        self.token_path = str(token_path)
        self.context_length = int(context_length)
        if self.context_length < 2:
            raise ValueError("context_length must be at least 2")
        size_bytes = __import__("os").path.getsize(self.token_path)
        if size_bytes % np.dtype(np.uint16).itemsize:
            raise ValueError("Token cache has an invalid uint16 byte length")
        self.token_count = size_bytes // np.dtype(np.uint16).itemsize
        if self.token_count <= self.context_length:
            raise ValueError("Token cache is too small for one context block")
        self._tokens = np.memmap(self.token_path, mode="r", dtype=np.uint16)

    def __len__(self):
        return (self.token_count - 1) // self.context_length

    def __getitem__(self, index):
        start = int(index) * self.context_length
        window = np.array(
            self._tokens[start:start + self.context_length + 1],
            dtype=np.int64,
            copy=True,
        )
        x = torch.from_numpy(window[:-1])
        y = torch.from_numpy(window[1:])
        return x, y
