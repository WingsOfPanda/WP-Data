import argparse
import logging
import os
import random
import sys
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np

from wputils.utils.io import rseq, wseq, rnpz
from wputils.utils.norm import nml


class BasicDataset(Dataset):
    def __init__(self, **kwargs):
        self.ids = kwargs.get('filepaths')
        self.pimg = kwargs.get('fpimg')
        self.plab = kwargs.get('fplab')
        self.pconfig = kwargs.get('fpconfig')
        if not self.ids:
            raise RuntimeError(f'No input file found in {self.ids}, make sure you put your images there')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        datadicts = rnpz(name)
        img = datadicts['image']
        lab = datadicts['label']

        assert img.size == lab.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {lab.size}'

        img = self.pimg(img, **self.pconfig)
        lab = self.plab(lab, **self.pconfig)

        return torch.as_tensor(img.copy()).float().contiguous(), torch.as_tensor(lab.copy()).float().contiguous()


class WPDataset(BasicDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # logging.info(f'Using device {device}')
    #
    # filedir = '/data/liupan/T2SegData/npzfile/original/validfiles'
    # dataset = T2Dataset(filedir)
    #
    # # Create a DataLoader instance
    # batch_size = 1
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    #
    # for epoch in range(16):
    #     for batch_idx, datadict in enumerate(data_loader):
    #         print(batch_idx, datadict['img'].shape, datadict['lab'].shape)
