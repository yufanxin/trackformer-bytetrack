"""
VisDrone Wrapper which combines sequences to a dataset.
"""
from torch.utils.data import Dataset

from .VisDrone_sequence import VisDroneSequence


class VisDroneWrapper(Dataset):
    def __init__(self, split: str, **kwargs) -> None:
        super().__init__()
        if split in ['train', 'test', 'val', 'all']:
            raise NotImplementedError

        self._data = []  # list stored the seq
        self._data.append(VisDroneSequence(seq_name=split, **kwargs))

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)