# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Factory of tracking datasets.
"""
from typing import Union

from torch.utils.data import ConcatDataset

from .mot_wrapper import MOT17Wrapper, MOTS20Wrapper
from .VisDrone_wrapper import VisDroneWrapper
from .demo_sequence import DemoSequence

import os

DATASETS = {}


def readVisDroneSeqs():
    """
    Read the VisDrone Seqs, including train, val and test.
    """
    VisDronePath = '/data/wujiapeng/datasets/VisDrone2019/VisDrone2019/'
    train_val_testList = ['VisDrone2019-MOT-train', 'VisDrone2019-MOT-val', 'VisDrone2019-MOT-test-dev']

    VisDroneSeqs = []

    for train_val_test_l in train_val_testList:
        path = os.path.join(VisDronePath, train_val_test_l, 'sequences')
        VisDroneSeqs.extend(os.listdir(path))

    return VisDroneSeqs


# Fill all available datasets, change here to modify / add new datasets.
'''
for split in ['TRAIN', 'TEST', 'ALL', '01', '02', '03', '04', '05',
              '06', '07', '08', '09', '10', '11', '12', '13', '14']:
    for dets in ['DPM', 'FRCNN', 'SDP', 'ALL']:
        name = f'MOT17-{split}'
        if dets:
            name = f"{name}-{dets}"
        DATASETS[name] = (
            lambda kwargs, split=split, dets=dets: MOT17Wrapper(split, dets, **kwargs)) # one name corrospond to one seq
'''
# Add VisDrone into dict(DATASETS)
# add DATASETS[name] = (lambda kwargs, name: VisDroneWrapper(name, **kwargs))

# TODO Firstly add sum seqs
'''
for split in ['train', 'test', 'val', 'all']:
    name = f'VisDrone-{split}'
    DATASETS[name] = (
        lambda kwargs, split=split: VisDroneWrapper(split=split, **kwargs))
'''
# Secondly add specific seqs

VisDroneSeqs = readVisDroneSeqs()
for split in VisDroneSeqs:
    name = split
    DATASETS[name] = (
        lambda kwargs, split=split: VisDroneWrapper(split=split, **kwargs))

########
'''
for split in ['TRAIN', 'TEST', 'ALL', '01', '02', '05', '06', '07', '09', '11', '12']:
    name = f'MOTS20-{split}'
    DATASETS[name] = (
        lambda kwargs, split=split: MOTS20Wrapper(split, **kwargs))
'''
DATASETS['DEMO'] = (lambda kwargs: [DemoSequence(**kwargs), ])


class TrackDatasetFactory:
    """A central class to manage the individual dataset loaders.

    This class contains the datasets. Once initialized the individual parts (e.g. sequences)
    can be accessed.
    """

    def __init__(self, datasets: Union[str, list], **kwargs) -> None:
        """Initialize the corresponding dataloader.

        Keyword arguments:
        datasets --  the name of the dataset or list of dataset names
        kwargs -- arguments used to call the datasets
        """
        if isinstance(datasets, str):
            datasets = [datasets]

        self._data = None
        for dataset in datasets:
            assert dataset in DATASETS, f"[!] Dataset not found: {dataset}"

            if self._data is None:
                self._data = DATASETS[dataset](kwargs)
            else:
                self._data = ConcatDataset([self._data, DATASETS[dataset](kwargs)])

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]
