# gaussian.py

import math
import pdb

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# from lmfit import Model

__all__ = ['Pulse_Actual_Double']


class PulsePrepare(pl.LightningDataModule):

    def __init__(self, data_dir, ratio_train, seed, centering_wid, train, val):

        self.data_dir = data_dir

        filename = [data_dir + '/real_double_traces.bin']
        data = []
        for k in range(len(filename)):
            i = 0
            with open(filename[k], 'rb') as fptr:
                while True:
                    size = np.fromfile(fptr, dtype='uint32', count=(1))
                    if not size.size:
                        break
                    one_sample = np.fromfile(fptr, dtype='uint16', count=(size[0]))
                    # data.append(one_sample/(one_sample.max().astype(np.uint16)))
                    if one_sample.max() < 16383:
                        data.append(one_sample)
                    i = i + 1
        # data = np.array(data, dtype='uint16')
        data = np.array(data)

        if centering_wid > 0:
        # ofs = np.mean(data_s[:, 0:centering_wid], axis=1)[:, np.newaxis] 
            ofs = np.partition(data, centering_wid, axis=1)[:, :centering_wid].mean(axis=1)[:, np.newaxis]
            data = data - ofs
        # pdb.set_trace()

        noise = [0, 10, 80, 115, 154, 163, 168, 172, 173, 175, 182]
        data = np.delete(data, noise, axis=0)
        # for k in range(len(data)):
        #     plt.plot(data[k])
        #     print('idx:', k)
        #     plt.show()

###################################################################

        if train:
            self.x1 = torch.from_numpy(data.astype(np.float32)).float()
            self.x2 = torch.from_numpy(data.astype(np.float32)).float()
            self.x = self.x1
            self.maxes1 = torch.max(self.x1, dim=1).values
            self.maxes2 = torch.max(self.x2, dim=1).values
            self.maxes = torch.max(self.x, dim=1).values
            self.dt = torch.zeros(len(self.x1))

        elif val:
            self.x1 = torch.from_numpy(data.astype(np.float32)).float()
            self.x2 = torch.from_numpy(data.astype(np.float32)).float()
            self.x = self.x1
            self.maxes1 = torch.max(self.x1, dim=1).values
            self.maxes2 = torch.max(self.x2, dim=1).values
            self.maxes = torch.max(self.x, dim=1).values
            self.dt = torch.zeros(len(self.x1))

        else:
            self.x1 = torch.from_numpy(data.astype(np.float32)).float()
            self.x2 = torch.from_numpy(data.astype(np.float32)).float()
            self.x = self.x1
            self.maxes1 = torch.max(self.x1, dim=1).values
            self.maxes2 = torch.max(self.x2, dim=1).values
            self.maxes = torch.max(self.x, dim=1).values
            self.dt = torch.zeros(len(self.x1))

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, index):
        x1, x2, x, m1, m2, m, dt = self.x1[index], self.x2[index], self.x[index], self.maxes1[index], self.maxes2[index], self.maxes[index], self.dt[index]
        return x1, x2, x, m1, m2, m, dt


class Pulse_Actual_Double(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.data_dir = opts.features_path
        self.ratio_train = opts.dataset_options['ratio_train']
        self.seed = opts.dataset_options['seed']
        self.centering_wid = opts.centering_wid

        # dataset = PulsePrepare(data_dir=self.data_dir, ratio_train=self.ratio_train, seed=self.seed,
                            #   normalize=self.normalize, train=True, val=False)
        # self.mean = dataset.mean
        # self.std = dataset.std
        # self.t = dataset.t

        if opts.ngpu == 0:
            self.pin_memory = False
        else:
            self.pin_memory = True

    def train_dataloader(self):
        batch_size = self.opts.batch_size_train
        dataset = PulsePrepare(data_dir=self.data_dir, ratio_train=self.ratio_train, seed=self.seed,
                              centering_wid=self.centering_wid, train=True, val=False)
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            # persistent_workers=True,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self):
        batch_size = self.opts.batch_size_val

        dataset = PulsePrepare(data_dir=self.data_dir, ratio_train=self.ratio_train, seed=self.seed,
                               centering_wid=self.centering_wid, train=False, val=True)

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            # persistent_workers=True,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self):
        batch_size = self.opts.batch_size_test

        dataset = PulsePrepare(data_dir=self.data_dir, ratio_train=self.ratio_train, seed=self.seed,
                               centering_wid=self.centering_wid, train=False, val=False)

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_train_dataloader(self):
        batch_size = self.opts.batch_size_test

        dataset = PulsePrepare(data_dir=self.data_dir, ratio_train=self.ratio_train, seed=self.seed,
                              train=True, val=False)

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader