
'''
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2025 <Your Name or Organization>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-- Bashir Sadeghi
'''

import pdb
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

__all__ = ['Pulse_SAEFit']

MAX_SHIFT=84
MAX_POS=155

class PulsePrepare(torch.utils.data.Dataset):

    def __init__(self, data_dir, ratio_train, ratio_test, seed, centering_wid, t_res, a_res, train, val):

        self.data_dir = data_dir
        filename = data_dir + '/real_single_traces.bin'
    
        data_s = []
        with open(filename, 'rb') as fptr:
            while True:
                size = np.fromfile(fptr, dtype='uint32', count=(1))
                if not size.size:
                    break
                one_sample = np.fromfile(fptr, dtype='uint16', count=(size[0]))
                if one_sample.max() < 16383: # remove saturated pulses
                    data_s.append(one_sample)
        data_s = np.array(data_s)

        # pdb.set_trace()

###################################################################
        ## Second pulse data generation from the first pulse data
        if centering_wid > 0:
            # ofs = np.mean(data_s[:, 0:centering_wid], axis=1)[:, np.newaxis] 
            ofs = np.partition(data_s, centering_wid, axis=1)[:, :centering_wid].mean(axis=1)[:, np.newaxis]
            data_s = data_s - ofs
        
        data_s_train, data_s_test = train_test_split(data_s, train_size=ratio_train, random_state=seed)
        data_s_test, data_s_val = train_test_split(data_s_test, train_size=ratio_test / (1 - ratio_train), random_state=seed)


        t1_shifts = [[-MAX_SHIFT, -MAX_SHIFT//2], [-MAX_SHIFT//2, 0], [0, MAX_SHIFT//2], [MAX_SHIFT//2, MAX_SHIFT]]
        t_diffs = [[2, t_res[0]+1]] + [[a+1, b+1] for a, b in zip(t_res[:-1], t_res[1:])]
        a_scales2 = [[a, b] for a, b in zip(a_res[:-1], a_res[1:])]
        a_scales1 = [[1/a, 1/b] for a, b in zip(a_res[::-1][:-1], a_res[::-1][1:])]
        a_scales = a_scales1 +[[1/a_res[0], 1], [1, a_res[0]]] + a_scales2
       
        nt_sample = data_s_train.shape[1]
        #################################### Shifting the first pulse #####################################
        datasets = {
            'tr': data_s_train,
            'val': data_s_val,
            'te': data_s_test
        }

        shifted1 = {
            'tr': [],
            'val': [],
            'te': []
        }
        shifted2 = {
            'tr': [],
            'val': [],
            'te': []
        }

        t1 = {
            'tr': [],
            'val': [],
            'te': []
        }

        dt_shift = {
            'tr': [],
            'val': [],
            'te': []
        }

        n_samples = {
            'tr': 3000,
            'val': 350,
            'te': 700
        }

        for name, data in datasets.items():
            j = 0
            l_data = len(data)
            for t1_s in t1_shifts:
                for t_d in t_diffs:
                    if t_d[1]+t1_s[1] <= MAX_POS:
                        for a_s in a_scales:

                            t1_shift = np.random.randint(t1_s[0], t1_s[1], size=(n_samples[name],))
                            t2_shift = np.random.randint(t_d[0], t_d[1], size=(n_samples[name],))
                            
                            start_idx = j * n_samples[name] % l_data
                            end_idx = (j + 1) * n_samples[name] % l_data
                            if start_idx < end_idx:
                                # Normal slicing
                                data_samples = data[start_idx:end_idx]
                            else:
                                # Wrap-around slicing
                                data_samples = np.concatenate((data[start_idx:], data[:end_idx]), axis=0)

                            a_scale = 10 ** np.random.uniform(np.log10(a_s[0]), np.log10(a_s[1]), size=(n_samples[name], 1))


                            shifted1_samples = np.zeros_like(data_samples)
                            for i, shift in enumerate(t1_shift):
                                shifted1_samples[i] = np.roll(data_samples[i], shift)
                                if shift < 0:
                                    shifted1_samples[i, nt_sample+shift:] = data_samples[i, -1]
                                else:
                                    shifted1_samples[i, :shift] = data_samples[i, 0]
                            
                            shifted2_samples = np.zeros_like(shifted1_samples)
                            for i, shift in enumerate(t2_shift):
                                shifted2_samples[i] = np.roll(shifted1_samples[i], shift)
                                shifted2_samples[i, :shift] = shifted1_samples[i, 0]
                            
                            shifted1[name].append(np.array(shifted1_samples * a_scale))
                            shifted2[name].append(np.array(shifted2_samples))
                            t1[name].append(np.array(t1_shift))
                            dt_shift[name].append(np.array(t2_shift))
                            j += 1

        data_train1 = np.concatenate(shifted1['tr'], axis=0)
        data_train2 = np.concatenate(shifted2['tr'], axis=0)

        data_val1 = np.concatenate(shifted1['val'], axis=0)
        data_val2 = np.concatenate(shifted2['val'], axis=0)

        data_test1 = np.concatenate(shifted1['te'], axis=0)
        data_test2 = np.concatenate(shifted2['te'], axis=0)

        dt_train = np.concatenate(dt_shift['tr'], axis=0)
        dt_val = np.concatenate(dt_shift['val'], axis=0)
        dt_test = np.concatenate(dt_shift['te'], axis=0)

###################################################################

        if train:
            self.x1 = torch.from_numpy(data_train1).float()
            self.x2 = torch.from_numpy(data_train2).float()
            self.x = self.x1 + self.x2
            self.maxes1 = torch.max(self.x1, dim=1).values
            self.maxes2 = torch.max(self.x2, dim=1).values
            self.maxes = torch.max(self.x, dim=1).values
            self.dt = torch.from_numpy(dt_train).float()

        elif val:
            # pdb.set_trace()
            self.x1 = torch.from_numpy(data_val1).float()
            self.x2 = torch.from_numpy(data_val2).float()
            self.x = self.x1 + self.x2
            self.maxes1 = torch.max(self.x1, dim=1).values
            self.maxes2 = torch.max(self.x2, dim=1).values
            self.maxes = torch.max(self.x, dim=1).values
            self.dt = torch.from_numpy(dt_val).float()
            # pdb.set_trace()

        else:
            self.x1 = torch.from_numpy(data_test1).float()
            self.x2 = torch.from_numpy(data_test2).float()
            self.x = self.x1 + self.x2
            self.maxes1 = torch.max(self.x1, dim=1).values
            self.maxes2 = torch.max(self.x2, dim=1).values
            self.maxes = torch.max(self.x, dim=1).values
            self.dt = torch.from_numpy(dt_test).float()

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, index):
        x1, x2, x, m1, m2, m, dt = self.x1[index], self.x2[index], self.x[index], self.maxes1[index], self.maxes2[index], self.maxes[index], self.dt[index]
        return x1, x2, x,  m1, m2, m, dt


class Pulse_SAEFit(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.data_dir = opts.features_path
        self.ratio_train = opts.dataset_options['ratio_train']
        self.ratio_test = opts.dataset_options['ratio_test']
        self.seed = opts.dataset_options['seed']
        # self.normalize = opts.dataset_options['normalize']
        self.centering_wid = opts.centering_wid
        self.scaling = opts.scaling
        self.t_res = opts.t_res
        self.a_res = opts.a_res


        if opts.ngpu == 0:
            self.pin_memory = False
        else:
            self.pin_memory = True

    def train_dataloader(self):
        batch_size = self.opts.batch_size_train
        dataset = PulsePrepare(data_dir=self.data_dir, ratio_train=self.ratio_train, ratio_test=self.ratio_test, seed=self.seed,
                              centering_wid=self.centering_wid, t_res=self.t_res, a_res=self.a_res, train=True, val=False)
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
        dataset = PulsePrepare(data_dir=self.data_dir, ratio_train=self.ratio_train, ratio_test=self.ratio_test, seed=self.seed,
                              centering_wid=self.centering_wid, t_res=self.t_res, a_res=self.a_res, train=False, val=True)
       
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
        dataset = PulsePrepare(data_dir=self.data_dir, ratio_train=self.ratio_train, ratio_test=self.ratio_test, seed=self.seed,
                            centering_wid=self.centering_wid, t_res=self.t_res, a_res=self.a_res, train=False, val=False)
        
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
        dataset = PulsePrepare(data_dir=self.data_dir, ratio_train=self.ratio_train, ratio_test=self.ratio_test, seed=self.seed,
                              centering_wid=self.centering_wid, t_res=self.t_res, a_res=self.a_res, train=True, val=False)
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader