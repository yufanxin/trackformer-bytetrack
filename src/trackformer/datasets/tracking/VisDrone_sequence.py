"""
VisDrone Seqs Dataset
"""
import configparser
import csv
import os
import os.path as osp
from argparse import Namespace
from typing import Optional, Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..coco import make_coco_transforms
from ..transforms import Compose


class VisDroneSequence(Dataset):
    data_folder = 'VisDrone2019'

    def __init__(self, root_dir: str = '/data/wujiapeng/datasets', seq_name: Optional[str] = None,
                 vis_threshold: float = 0.0, img_transform: Namespace = None) -> None:

        super().__init__()

        self._seq_name = seq_name
        self._vis_threshold = vis_threshold
        root_dir = '/data/wujiapeng/datasets'  # Edited
        self._data_dir = osp.join(root_dir, self.data_folder,
                                  self.data_folder)  # /data/wujiapeng/datasets/VisDrone2019/VisDrone2019/

        # read the train, val and test seqs
        self._train_folders = os.listdir(os.path.join(self._data_dir, 'VisDrone2019-MOT-train', 'sequences'))
        # /data/wujiapeng/datasets/VisDrone2019/VisDrone2019/VisDrone2019-MOT-train/sequences
        self._test_folders = os.listdir(os.path.join(self._data_dir, 'VisDrone2019-MOT-test-dev', 'sequences'))
        self._val_folders = os.listdir(os.path.join(self._data_dir, 'VisDrone2019-MOT-val', 'sequences'))

        # img transform
        self.transforms = Compose(make_coco_transforms('val', img_transform))  # val is because only do random resize

        self.data = []
        self.no_gt = True

        if seq_name:
            assert seq_name in self._train_folders or seq_name in self._test_folders or seq_name in self._val_folders

        self.data = self._sequence()
        self.no_gt = not osp.exists(self.get_gt_file_path())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return the ith image converted to blob
        """
        data = self.data[idx]  # data from idx th image
        img = Image.open(data['im_path']).convert("RGB")
        width_orig, height_orig = img.size

        img, _ = self.transforms(img)  # do transforms
        width, height = img.size(2), img.size(1)  # size after transforms

        sample = {}  # initialize as dict to store blob
        sample['img'] = img  # img after transforms
        sample['img_path'] = data['im_path']  # img path
        sample['gt'] = data['gt']
        sample['vis'] = data['vis']
        sample['dets'] = torch.tensor([])  # must have 'dets' key
        # size before transform
        sample['orig_size'] = torch.as_tensor([int(height_orig), int(width_orig)])
        # size after transform
        sample['size'] = torch.as_tensor([int(height), int(width)])

        return sample

    def get_seq_path(self):
        """
        get full path for certain seq
        """
        seq_name = self._seq_name
        if seq_name in self._train_folders:
            return osp.join(self._data_dir, 'VisDrone2019-MOT-train')
            # /data/wujiapeng/datasets/VisDrone2019/VisDrone2019/VisDrone2019-MOT-train/
        elif seq_name in self._test_folders:
            return osp.join(self._data_dir, 'VisDrone2019-MOT-test-dev')
        else:
            return osp.join(self._data_dir, 'VisDrone2019-MOT-val')

    def get_gt_file_path(self):
        """
        return GT file of seqs
        """
        # /data/wujiapeng/datasets/VisDrone2019/VisDrone2019/VisDrone2019-MOT-train/
        # /annotations/uav0000076_00720_v.txt
        return osp.join(self.get_seq_path(), 'annotations', self._seq_name + '.txt')

    def get_track_boxes_and_visbility(self):
        """
        load GT boxes and visibility
        """
        boxes, visibility = {}, {}
        print(f"****seq_length:{self.seq_length}****")
        # create boxes and visibility for each image in seq
        for i in range(1, self.seq_length + 1):
            boxes[i] = {}
            visibility[i] = {}

        gt_file = self.get_gt_file_path()
        if not osp.exists(gt_file):
            return boxes, visibility

        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=',')
            for row in reader:
                # class person, certainty 1
                # if int(row[6]) == 1 and int(row[7]) == 4:
                if int(row[6]) == 1 and int(row[7]) == 4:
                    # Make pixel indexes 0-based, should already be 0-based (or not)
                    x1 = int(row[2]) - 1
                    y1 = int(row[3]) - 1
                    # This -1 accounts for the width (width of 1 x1=x2)
                    x2 = x1 + int(row[4]) - 1
                    y2 = y1 + int(row[5]) - 1
                    bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

                    frame_id = int(row[0])
                    track_id = int(row[1])

                    boxes[frame_id][track_id] = bbox  # note the bbox of track_id in frame
                    visibility[frame_id][track_id] = float(row[8])  # note the visibility of track_id in frame

        return boxes, visibility

    def _sequence(self):
        """
        note the data of this seq
        Attn: Different with MOT, no det files, use gt files instead.
        """
        boxes, visibility = self.get_track_boxes_and_visbility()

        img_dir = osp.join(self.get_seq_path(), 'sequences', self._seq_name)
        # /data/wujiapeng/datasets/VisDrone2019/VisDrone2019/VisDrone2019-MOT-train/sequences/seq_name
        total = [
            {'gt': boxes[i],
             'im_path': osp.join(img_dir, f"{i:07d}.jpg"),
             'vis': visibility[i]}
            for i in range(1, self.seq_length + 1)]  # No key named det like MOT17p

        return total

    @property
    def seq_length(self):
        """
        return num of images(frames) in seq
        """

        seq = os.listdir(osp.join(self.get_seq_path(), 'sequences', self._seq_name))
        # print(seq)
        return len(seq)

    # TODO Lack codes of writing results

    @property
    def results_file_name(self):
        assert self._seq_name is not None

        return f"{self}.txt"

    def load_results(self, results_dir):
        results = {}
        if results_dir is None:
            return results

        return {}
