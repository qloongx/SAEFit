
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

import torch
import pytorch_lightning as pl
import os
import csv

import lab.models as models
import lab.losses as losses
import lab.metrics as metrics
import lab.schedulers as schedulers_custom
import matplotlib.pyplot as plt

import numpy as np

import pdb


__all__ = ['SAEFit']


class SAEFit(pl.LightningModule):
    def __init__(self, opts, dataloader):
        super().__init__()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        self.data_flag = None
        self.save_hyperparameters()

        self.opts = opts
        self.r = opts.r
        self.best_loss = 10.0
        self.n_t_samples = 250

        ## Logging Separatly in CSV
        # self.loss_log_file = 'epoch_metrics_train.csv'
        self.csv_initialized = False  # flag to prevent reinitializing header each epoch

        #################################################################################
        #                                  Models                                       #
        #################################################################################
        self.model = getattr(models, opts.model_type)(**opts.model_options)

        #################################################################################
        #                          Losses and Accuracies                                #
        #################################################################################
        self.criterion = {}

        self.criterion['train_lossps'] = getattr(losses, opts.loss_type)(**opts.loss_options)

        self.criterion['train_lossp1'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['train_lossp2'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['train_losspd'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['train_lossrd'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['train_lossr1'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['train_losst'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['train_lossc'] = getattr(losses, opts.loss_type2)(**opts.loss_options2)

        self.criterion['val_lossps'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['val_lossp1'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['val_lossp2'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['val_losspd'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['val_lossrd'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['val_lossr1'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['val_losst'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['val_lossc'] = getattr(losses, opts.loss_type2)(**opts.loss_options2)

        self.criterion['test_lossps'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['test_lossp1'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['test_lossp2'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['test_losspd'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['test_lossrd'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['test_lossr1'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['test_losst'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['test_lossc'] = getattr(losses, opts.loss_type2)(**opts.loss_options2)

        self.acc_trn = getattr(metrics, opts.evaluation_type_1)(**opts.evaluation_options_1)
        self.snr_ps_trn = getattr(metrics, opts.evaluation_type_2)(**opts.evaluation_options_2)
        self.snr_p1_trn = getattr(metrics, opts.evaluation_type_2)(**opts.evaluation_options_2)
        self.snr_p2_trn = getattr(metrics, opts.evaluation_type_2)(**opts.evaluation_options_2)
        self.snr_pd_trn = getattr(metrics, opts.evaluation_type_2)(**opts.evaluation_options_2)
        self.cm_trn = getattr(metrics, opts.evaluation_type_3)(**opts.evaluation_options_3)

        self.acc_val = getattr(metrics, opts.evaluation_type_1)(**opts.evaluation_options_1)
        self.snr_ps_val = getattr(metrics, opts.evaluation_type_2)(**opts.evaluation_options_2)
        self.snr_p1_val = getattr(metrics, opts.evaluation_type_2)(**opts.evaluation_options_2)
        self.snr_p2_val = getattr(metrics, opts.evaluation_type_2)(**opts.evaluation_options_2)
        self.snr_pd_val = getattr(metrics, opts.evaluation_type_2)(**opts.evaluation_options_2)
        self.cm_val = getattr(metrics, opts.evaluation_type_3)(**opts.evaluation_options_3)

        self.acc_test = getattr(metrics, opts.evaluation_type_1)(**opts.evaluation_options_1)
        self.snr_ps_test = getattr(metrics, opts.evaluation_type_2)(**opts.evaluation_options_2)
        self.snr_p1_test = getattr(metrics, opts.evaluation_type_2)(**opts.evaluation_options_2)
        self.snr_p2_test = getattr(metrics, opts.evaluation_type_2)(**opts.evaluation_options_2)
        self.snr_pd_test = getattr(metrics, opts.evaluation_type_2)(**opts.evaluation_options_2)
        self.cm_test = getattr(metrics, opts.evaluation_type_3)(**opts.evaluation_options_3)

        self.acc_tst = {}
        self.snr_ps_tst = {}
        self.snr_p1_tst = {}
        self.snr_p2_tst = {}
        self.snr_pd_tst = {}
        self.cm_tst = {}
        self.amplt_f_tst = {}
        self.amplt_s_tst = {}
        self.dt_tst = {}


        ################################################# Training ########################################
    def training_step(self, batch, batch_idx):
        x1, x2, xd, m1, m2, m, dt = batch

        # These are normalized subpulses as a part of double pulse that we use for training
        x1N = x1 / m[:, None]
        x2N = x2 / m[:, None]

        # These are normalized subpulses as individual pulses that we use for evaluation
        x1 = x1 / m1[:, None]
        x2 = x2 / m2[:, None]

        xd = xd / m[:, None]

        batch_size = len(x1)
        opt = self.optimizers()

        control = False
        if self.current_epoch < self.opts.init_epochs:
            control = True

        outs, preds, ts, a_s, _, ks, rs, _, _, _,_ = self.model(x1, control=control)

        outd, predd, t1, _, a1, k1, r1, t2, a2, k2, r2 = self.model(xd, control=control)

        t = (torch.arange(x1.shape[1], device=x1.device).unsqueeze(0) / self.n_t_samples).detach()
        dt = dt.unsqueeze(1).detach() / self.n_t_samples

        ps = self.logistic_function(t, ts, a_s, ks, rs)

        p1 = self.logistic_function(t, t1, a1, k1, r1)

        p2 = self.logistic_function(t, t2, a2, k2, r2)

        pd = p1 + p2

        y = torch.cat((0*torch.ones(batch_size,), 1*torch.ones(batch_size,))).to(device=x1.device).long()
        pred0 = torch.cat((preds, predd), dim=0)

        pred = torch.argmax(pred0, dim=1)
        self.acc_trn.update(pred, y)
        self.cm_trn.update(pred, y)

        loss_c = self.criterion['train_lossc'](pred0, y)
        loss_rs = self.criterion['train_lossr1'](outs, x1)
        loss_rd = self.criterion['train_lossrd'](outd, xd)

        if self.current_epoch < self.opts.init_epochs:
            loss_ps = self.criterion['train_lossps'](ps.expand(x1.shape), x1)
            loss_p1 = self.criterion['train_lossp1'](p1.expand(x1N.shape), x1N)
            loss_p2 = self.criterion['train_lossp2'](p2.expand(x2N.shape), x2N)
            loss_pd = self.criterion['train_losspd'](pd.expand(xd.shape), xd)
            loss_t = self.criterion['train_losst']((t2-t1).expand(dt.shape), dt)

            self.snr_ps_trn.update(ps, x1.mean(0, True))
            self.snr_pd_trn.update(pd, xd.mean(0, True))
            self.snr_p1_trn.update(p1, x1N.mean(0, True))
            self.snr_p2_trn.update(p2, x2N.mean(0, True))

        else:
            loss_ps = self.criterion['train_lossps'](ps, x1)
            loss_p1 = self.criterion['train_lossp1'](p1, x1N)
            loss_p2 = self.criterion['train_lossp2'](p2, x2N)
            loss_pd = self.criterion['train_losspd'](pd, xd)
            loss_t = self.criterion['train_losst'](t2-t1, dt)

            self.snr_ps_trn.update(ps, x1)
            self.snr_pd_trn.update(pd, xd)
            self.snr_p1_trn.update(p1, x1N)
            self.snr_p2_trn.update(p2, x2N)

        loss = (loss_pd + loss_ps + loss_p1 + loss_p2) \
        + self.opts.lam * (loss_rs + loss_rd) \
        + self.opts.tauc * loss_c \
        + self.opts.tau * loss_t

        self.manual_backward(loss)

        with torch.no_grad():
            for param in self.model.parameters():
                # pdb.set_trace()
                if param.grad is not None:
                    nan_mask = torch.isnan(param.grad)
                    param.grad[nan_mask] = 0.0

        opt.step()
        opt.zero_grad()

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        sch.step()

        loss_ps = self.criterion['train_lossps'].compute()

        loss_p1 = self.criterion['train_lossp1'].compute()
        loss_p2 = self.criterion['train_lossp2'].compute()
        loss_pd = self.criterion['train_losspd'].compute()

        loss_c = self.criterion['train_lossc'].compute()

        loss_rs = self.criterion['train_lossr1'].compute()
        loss_rd = self.criterion['train_lossrd'].compute()

        loss_t = self.criterion['train_losst'].compute()


        snr_ps = self.snr_ps_trn.compute()
        snr_pd = self.snr_pd_trn.compute()
        snr_p1 = self.snr_p1_trn.compute()
        snr_p2 = self.snr_p2_trn.compute()

        acc = self.acc_trn.compute() * 100
        cm = self.cm_trn.compute()
        cm_percentage = cm.float() / cm.sum(axis=1, keepdim=True) * 100
        ts, fd, fs, td = cm_percentage.ravel()

        self.log('train_loss_ps', loss_ps, on_epoch=True, prog_bar=True)

        self.log('train_loss_p1', loss_p1, on_epoch=True, prog_bar=False)
        self.log('train_loss_p2', loss_p2, on_epoch=True, prog_bar=False)
        self.log('train_loss_pd', loss_pd, on_epoch=True, prog_bar=True)

        self.log('train_loss_c', loss_c, on_epoch=True, prog_bar=False)
        self.log('train_loss_rs', loss_rs, on_epoch=True, prog_bar=False)
        self.log('train_loss_rd', loss_rd, on_epoch=True, prog_bar=False)
        self.log('train_loss_dt', loss_t, on_epoch=True, prog_bar=False)

        self.log('train_snr_ps', snr_ps, on_epoch=True, prog_bar=False)
        self.log('train_snr_pd', snr_pd, on_epoch=True, prog_bar=False)
        self.log('train_snr_p1', snr_p1, on_epoch=True, prog_bar=False)
        self.log('train_snr_p2', snr_p2, on_epoch=True, prog_bar=False)

        self.log('train_acc', acc, on_epoch=True, prog_bar=True)
        self.log('train_true_single', ts, on_epoch=True, prog_bar=False)
        self.log('train_false_double', fd, on_epoch=True, prog_bar=False)
        self.log('train_false_single', fs, on_epoch=True, prog_bar=False)
        self.log('train_true_double', td, on_epoch=True, prog_bar=False)

        self.criterion['train_lossps'].reset()

        self.criterion['train_lossp1'].reset()
        self.criterion['train_lossp2'].reset()
        self.criterion['train_losspd'].reset()
        self.criterion['train_lossc'].reset()
        self.criterion['train_lossr1'].reset()
        self.criterion['train_lossrd'].reset()
        self.criterion['train_losst'].reset()
        # self.criterion['train_losst0'].reset()

        self.snr_ps_trn.reset()
        self.snr_pd_trn.reset()
        self.snr_p1_trn.reset()
        self.snr_p2_trn.reset()
        self.acc_trn.reset()
        self.cm_trn.reset()
        # torch.cuda.empty_cache()

        # print('learning rate:' ,self.opts.learning_rate)
        ###### Logging Separately in CSV
        pulse_dir = os.path.join(self.opts.out_dir, 'pulses')
        try:
            if not os.path.exists(pulse_dir):
                os.makedirs(pulse_dir)
        except:
            pass
        csv_path = os.path.join(pulse_dir, 'epoch_metrics_train.csv')
        if not self.csv_initialized:
            if not os.path.exists(csv_path):
                with open(csv_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'epoch', 'loss_ps', 'loss_p1', 'loss_p2', 'loss_pd',
                        'loss_c', 'loss_rd', 'loss_t',
                        'snr_ps', 'snr_pd', 'snr_p1', 'snr_p2', 'acc',
                        'ts', 'fd', 'fs', 'td'
                    ])
        self.csv_initialized = True

        
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_epoch,
                loss_ps.item(), loss_p1.item(), loss_p2.item(), loss_pd.item(),
                loss_c.item(), loss_rd.item(), loss_t.item(),
                snr_ps.item(), snr_pd.item(), snr_p1.item(), snr_p2.item(), acc.item(),
                ts.item(), fd.item(), fs.item(), td.item()
            ])

    ################################ Validation #######################################################################
    def validation_step(self, batch, batch_idx):
        x1, x2, xd, m1, m2, m, dt = batch
        # x = x1 + x2

        x1N = x1 / m[:, None]
        x2N = x2 / m[:, None]
        x1 = x1 / m1[:, None]
        x2 = x2 / m2[:, None]
        xd = xd / m[:, None]

        batch_size = len(x1)

        control = False
        if self.current_epoch < self.opts.init_epochs:
            control = True

        # [t1, a1s, a1, k1, k2, ofs1, t2, a2, k3, k4, ofs2]
        _, preds, ts, a_s, _, ks, rs, _, _, _, _ = self.model(x1,control=control)

        # output = self.model(x, control=control)
        outd, predd, t1, _, a1, k1, r1, t2, a2, k2, r2 = self.model(xd, control=control)

        t = (torch.arange(x1.shape[1], device=x1.device).unsqueeze(0) / self.n_t_samples).detach()
        dt = dt.unsqueeze(1).detach() / self.n_t_samples
        ps = self.logistic_function(t, ts, a_s, ks, rs)
        p1 = self.logistic_function(t, t1, a1, k1, r1)
        p2 = self.logistic_function(t, t2, a2, k2, r2)

        pd = p1 + p2
        # p = torch.clamp(pf + ps, min=0, max=5)

        y = torch.cat((0*torch.ones(batch_size,), 1*torch.ones(batch_size,))).to(device=x1.device).long()
        pred0 = torch.cat((preds, predd), dim=0)
        pred = torch.argmax(pred0, dim=1)
        self.acc_val.update(pred, y)
        self.cm_val.update(pred, y)

        if self.current_epoch < self.opts.init_epochs:
            loss_ps = self.criterion['val_lossps'](ps.expand(x1.shape), x1)
            loss_p1 = self.criterion['val_lossp1'](p1.expand(x1N.shape), x1N)
            loss_p2 = self.criterion['val_lossp2'](p2.expand(x2N.shape), x2N)
            loss_pd = self.criterion['val_losspd'](pd.expand(xd.shape), xd)
            loss_t = self.criterion['val_losst']((t2-t1).expand(dt.shape), dt)

            self.snr_ps_val.update(ps, x1.mean(0, True))
            self.snr_pd_val.update(pd, xd.mean(0, True))
            self.snr_p1_val.update(p1, x1N.mean(0, True))
            self.snr_p2_val.update(p2, x2N.mean(0, True))
        else:
            loss_ps = self.criterion['val_lossps'](ps, x1)
            loss_p1 = self.criterion['val_lossp1'](p1, x1N)
            loss_p2 = self.criterion['val_lossp2'](p2, x2N)
            loss_pd = self.criterion['val_losspd'](pd, xd)
            loss_t = self.criterion['val_losst'](t2-t1, dt)

            self.snr_ps_val.update(ps, x1)
            self.snr_pd_val.update(pd, xd)
            self.snr_p1_val.update(p1, x1N)
            self.snr_p2_val.update(p2, x2N)

        loss_c = self.criterion['val_lossc'](pred0, y)
        loss_rd = self.criterion['val_lossrd'](outd, xd)
        # self.norm_a21 = torch.linalg.norm(a21) / batch_size

        
        ########################################## Tensorboard #########################################
        ps  = ps * m1[:, None]
        x1 = x1 * m1[:, None]

        p1 = p1 * m[:, None]
        x1N = x1N * m[:, None]
        p2 = p2 * m[:, None]
        x2N = x2N * m[:, None]
        pd = pd * m[:, None]
        xd = xd * m[:, None]

        # pdb.set_trace()

        t = np.arange(x1.shape[1])
        f_s = 18
        l_w = 1.5
        if self.current_epoch < self.opts.init_epochs:
            idx = 0
            num = 1
        else:
            idx0 = []
            i = 0
            for k in range(batch_size):
                # if t2[k] - t1[k] < 0.02:
                if t2[k] - t1[k] < 0.04 and i < 10:
                # if pred0[k] != y[k] or pred0[k + len(t11)] != y[k + len(t11)]:
                    idx0.append(k)
                    i += 1
            num = 20 - len(idx0)
            idx1 = torch.randint(0, batch_size, (num,))
            if len(idx0) == 0:
                idx = idx1.long()
            else:
                idx = torch.cat((idx1, torch.tensor(idx0))).long()
            # idx = torch.tensor(idx0).long()
            num = len(idx)
            # pdb.set_trace()
        ############################################
        for i in range(num):
            if self.current_epoch < self.opts.init_epochs:
                count = 0
            else:
                count = idx[i].item()
            # pdb.set_trace()
            ############################################
            fig, ax = plt.subplots(figsize=(13, 8))
            ax.step(t, pd[count].cpu().numpy(), label='double_fit', linewidth=l_w)
            ax.step(t, xd[count].cpu().numpy(), label='double_data', linewidth=l_w)
            
            ax.set_title(  f'epoch={self.current_epoch}, '
                        f'dt={self.n_t_samples * dt[count].item():.0f}\n'
                        f't1={self.n_t_samples * t1[count].item():.3f}, t1m={self.n_t_samples * t1.mean().item():.3f}+{self.n_t_samples * t1.std().item():.3f}, a1={a1[count].item():.2f}, '
                        f'k1={k1.mean().item():.2f}+{k1.std().item():.2f}, r1={r1.mean().item():.2f}+{r1.std().item():.2f}\n'

                        f't2={self.n_t_samples * t2[count].item():.3f}, t2m={self.n_t_samples * t2.mean().item():.3f}+{self.n_t_samples * t2.std().item():.3f}, a2={a2[count].item():.2f}, '
                        f'k2={k2.mean().item():.2f}+{k2.std().item():.2f}, r2={r2.mean().item():.2f}+{r2.std().item():.2f}\n'
                        f'y[k]={y[count].item()}, pred[k]={pred[count].item()}, y[k+n]={y[count+batch_size].item()}, pred[k+n]={pred[count+batch_size].item()}'
                        )

            ax.set_xlabel('Time', fontsize=f_s)
            ax.set_ylabel('Voltage', fontsize=f_s)
            ax.tick_params(axis='both', which='major', labelsize=f_s)
            ax.legend(fontsize=f_s)
            tensorboard_v = self.logger.experiment
            tensorboard_v.add_figure(
                f'datapd {i}', fig, self.current_epoch
            )
            plt.close(fig)
            # pdb.set_trace()
            ############################################
            fig, ax = plt.subplots(figsize=(13, 8))
            
            ax.step(t, p1[count].cpu().numpy(), label='first_fit', linewidth=l_w)
            ax.step(t, x1N[count].cpu().numpy(), label='first_data', linewidth=l_w)
            ax.step(t, p2[count].cpu().numpy(), label='second_fit', linewidth=l_w)
            ax.step(t, x2N[count].cpu().numpy(), label='second_data', linewidth=l_w)
            ax.set_title(  f'epoch={self.current_epoch}\n'
                        f't1={t1[count].item():.3f}, t1m={t1.mean().item():.3f}+{t1.std().item():.3f}, a1={a1[count].item():.2f}, '
                        f'k1={k1.mean().item():.2f}+{k1.std().item():.2f}, r1={r1.mean().item():.2f}+{r1.std().item():.2f}\n'

                        f't2={t2[count].item():.3f}, t2m={t2.mean().item():.3f}+{t2.std().item():.3f}, a2={a2[count].item():.2f}, '
                        f'k2={k2.mean().item():.2f}+{k2.std().item():.2f}, r2={r2.mean().item():.2f}+{r2.std().item():.2f}\n'
                        f'y[k]={y[count].item()}, pred[k]={pred[count].item()}, y[k+n]={y[count+batch_size].item()}, pred[k+n]={pred[count+batch_size].item()}'
                        )

            ax.set_xlabel('Time', fontsize=f_s)
            ax.set_ylabel('Voltage', fontsize=f_s)
            ax.tick_params(axis='both', which='major', labelsize=f_s)
            ax.legend(fontsize=f_s)
            tensorboard_v = self.logger.experiment
            tensorboard_v.add_figure(
                f'datap12 {i}', fig, self.current_epoch
            )
            plt.close(fig)
            ############################################     
            fig, ax = plt.subplots(figsize=(13, 8))
            # pdb.set_trace()
            ax.step(t, ps[count].cpu().numpy(), label='single_fit', linewidth=l_w)
            ax.step(t, x1[count].cpu().numpy(), label='single_data', linewidth=l_w)
            ax.set_title(  f'epoch={self.current_epoch}\n'
                        f'ts={ts[count].item():.3f}, tsm={ts.mean().item():.3f}+{ts.std().item():.3f}, a_s={a_s[count].item():.2f}, '
                        f'ks={ks.mean().item():.2f}+{ks.std().item():.2f}, rs={rs.mean().item():.2f}+{rs.std().item():.2f}'
                        )


            ax.set_xlabel('Time', fontsize=f_s)
            ax.set_ylabel('Voltage', fontsize=f_s)
            ax.tick_params(axis='both', which='major', labelsize=f_s)
            ax.legend(fontsize=f_s)
            tensorboard_v = self.logger.experiment
            tensorboard_v.add_figure(
                f'dataps {i}', fig, self.current_epoch
            )
            plt.close(fig)
        # ############################################

        fig, ax = plt.subplots(figsize=(13, 8))
        ax.step(t, outd[0].cpu().numpy(), label='reconstructed data', linewidth=l_w)
        ax.set_title('reconstructed data')
        ax.set_xlabel('Time', fontsize=f_s)
        ax.set_ylabel('Voltage', fontsize=f_s)
        tensorboard_v = self.logger.experiment
        tensorboard_v.add_figure(
            'reconstructed data', fig, self.current_epoch
        )
        plt.close(fig)
        # del t
        # torch.cuda.empty_cache()

    def on_validation_epoch_end(self):

        loss_ps = self.criterion['val_lossps'].compute()
        loss_p1 = self.criterion['val_lossp1'].compute()
        loss_p2 = self.criterion['val_lossp2'].compute()
        loss_pd = self.criterion['val_losspd'].compute()
        loss_c = self.criterion['val_lossc'].compute()
        loss_rd = self.criterion['val_lossrd'].compute()
        loss_t = self.criterion['val_losst'].compute()

        acc = self.acc_val.compute() * 100
        cm = self.cm_val.compute()
        cm_percentage = cm.float() / cm.sum(axis=1, keepdim=True) * 100
        ts, fd, fs, td = cm_percentage.ravel()

        snr_ps = self.snr_ps_val.compute()
        snr_pd = self.snr_pd_val.compute()
        snr_p1 = self.snr_p1_val.compute()
        snr_p2 = self.snr_p2_val.compute()

        self.log('val_loss_ps', loss_ps, on_epoch=True, prog_bar=False)
        self.log('val_loss_p1', loss_p1, on_epoch=True, prog_bar=True)
        self.log('val_loss_p2', loss_p2, on_epoch=True, prog_bar=True)
        self.log('val_loss_pd', loss_pd, on_epoch=True, prog_bar=False)
        self.log('val_loss_c', loss_c, on_epoch=True, prog_bar=False)
        self.log('val_loss_rd', loss_rd, on_epoch=True, prog_bar=False)
        self.log('val_loss_dt', loss_t, on_epoch=True, prog_bar=False)

        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_true_single', ts, on_epoch=True, prog_bar=False)
        self.log('val_false_double', fd, on_epoch=True, prog_bar=False)
        self.log('val_false_single', fs, on_epoch=True, prog_bar=False)
        self.log('val_true_double', td, on_epoch=True, prog_bar=False)

        self.log('val_snr_pd', snr_pd, on_epoch=True, prog_bar=False)
        self.log('val_snr_ps', snr_ps, on_epoch=True, prog_bar=False)
        self.log('val_snr_p1', snr_p1, on_epoch=True, prog_bar=False)
        self.log('val_snr_p2', snr_p2, on_epoch=True, prog_bar=False)

        self.criterion['val_lossps'].reset()
        self.criterion['val_lossp1'].reset()
        self.criterion['val_lossp2'].reset()
        self.criterion['val_losspd'].reset()
        self.criterion['val_lossc'].reset()
        self.criterion['val_lossrd'].reset()
        self.criterion['val_losst'].reset()

        self.acc_val.reset()
        self.snr_ps_val.reset()
        self.snr_pd_val.reset()
        self.snr_p1_val.reset()
        self.snr_p2_val.reset()
        self.cm_val.reset()
        
        loss = loss_pd

        self.log('val_loss', loss, on_epoch=True, prog_bar=False)


    ################################ Testing ##########################################################################
    
    def test_step(self, batch, batch_idx):
        x1, x2, xd, m1, m2, m, dt = batch

        x1N = x1 / m[:, None]
        x2N = x2 / m[:, None]
        x1 = x1 / m1[:, None]
        x2 = x2 / m2[:, None]
        xd = xd / m[:, None]
        batch_size = len(x1)

        _, preds, ts, a_s, _, ks, rs = self.model(x1, control=False)[0:7]

        # output = self.model(x, control=control)
        outd, predd, t1, _, a1, k1, r1, t2, a2, k2, r2 = self.model(xd, control=False)

        t = (torch.arange(x1.shape[1], device=ts.device).unsqueeze(0) / self.n_t_samples).detach()
        

        ps = self.logistic_function(t, ts, a_s, ks, rs)

        p1 = self.logistic_function(t, t1, a1, k1, r1)
        p2 = self.logistic_function(t, t2, a2, k2, r2)

        pd = p1 + p2

        y = torch.cat((0*torch.ones(batch_size,), 1*torch.ones(batch_size,))).to(device=x1.device).long()
        pred0 = torch.cat((preds, predd), dim=0)
        probs_s = torch.nn.functional.softmax(preds, dim=1)
        probs_d = torch.nn.functional.softmax(predd, dim=1)
        pred = torch.argmax(pred0, dim=1)

        self.acc_test.update(pred, y)
        self.cm_test.update(pred, y)

        # pdb.set_trace()
        max_x1N = x1N.max(dim=1)[0]
        max_x2N = x2N.max(dim=1)[0]

        dt = dt.unsqueeze(1).detach() / self.n_t_samples
        loss_ps = self.criterion['test_lossps'](ps, x1)
        loss_c = self.criterion['test_lossc'](pred0, y)
        loss_p1 = self.criterion['test_lossp1'](p1, x1N)
        loss_p2 = self.criterion['test_lossp2'](p2, x2N)
        loss_pd = self.criterion['test_losspd'](pd, xd)
        loss_rd = self.criterion['test_lossrd'](outd, xd)
        loss_t = self.criterion['test_losst'](t2-t1, dt)

        self.snr_ps_test.update(ps, x1)
        self.snr_pd_test.update(pd, xd)
        self.snr_p1_test.update(p1, x1N)
        self.snr_p2_test.update(p2, x2N)


        ########################################## Tensorboard #########################################
        idx0 = []
        i = 0
        for k in range(batch_size):
            # if t2[k] - t1[k] < 0.02:
            # if t2[k] - t1[k] < 0.04 and i < 75:
            dtk = dt[k].item()
            peak_ratio = max_x1N[k].item() / max_x2N[k].item()

            if dtk >= 0.001 and dtk < 0.04 and i < 50:
                idx0.append(k)
                i += 1
            if dtk >= 0.04 and peak_ratio < 2 and peak_ratio > 0.5 and i >= 50 and i < 100:
                idx0.append(k)
                i += 1

        num = 50
        idx1 = torch.randint(0, batch_size, (num,))
        
        if len(idx0) > 0:
            idx = torch.cat((torch.tensor(idx0), idx1)).long()
        else:
            idx = idx1
        
        num = len(idx)

        ############################################
        ps  = ps * m1[:, None]
        x1 = x1 * m1[:, None]

        p1 = p1 * m[:, None]
        x1N = x1N * m[:, None]
        p2 = p2 * m[:, None]
        x2N = x2N * m[:, None]
        pd = pd * m[:, None]
        xd = xd * m[:, None]

        f_s = 18
        l_w = 1.5
        t = np.arange(x1.shape[1])
        # print('len data:', len(x1))
        pulse_dir = os.path.join(self.opts.out_dir, 'pulses')
        try:
            if not os.path.exists(pulse_dir):
                os.makedirs(pulse_dir)
        except:
            pass
        
        for i in range(num):

            count = idx[i].item()
            ############################################
   
            fig, ax = plt.subplots(figsize=(13, 8))
            ax.step(t, pd[count].cpu().numpy(), label='double_fit', linewidth=l_w)
            # ax.step(t, p_chi2_2, label='Chi2', linewidth=l_w)
            ax.step(t, xd[count].cpu().numpy(), label='double_data', linewidth=l_w)
            
            ax.set_title(  f'epoch={self.current_epoch},'
                        f'dt={self.n_t_samples*dt[count].item():.3f}\n'

                        f't1={self.n_t_samples*t1[count].item():.3f}, t1m={self.n_t_samples*t1.mean().item():.3f}+{self.n_t_samples*t1.std().item():.3f}, a1={a1[count].item():.2f}, '
                        f'k1={k1.mean().item():.2f}+{k1.std().item():.2f}, r1={r1.mean().item():.2f}+{r1.std().item():.2f}\n'

                        f't2={self.n_t_samples*t2[count].item():.3f}, t2m={self.n_t_samples*t2.mean().item():.3f}+{self.n_t_samples*t2.std().item():.3f}, a2={a2[count].item():.2f}, '
                        f'k2={k2.mean().item():.2f}+{k2.std().item():.2f}, r2={r2.mean().item():.2f}+{r2.std().item():.2f}\n'
                        f'y[k]={y[count].item()}, pred[k]={pred[count].item()}, y[k+n]={y[count+batch_size].item()}, pred[k+n]={pred[count+batch_size].item()}, '
                        f'pred0[k+n]={probs_d[count].detach().cpu().numpy()}'
                        )

            ax.set_xlabel('Time', fontsize=f_s)
            ax.set_ylabel('Voltage', fontsize=f_s)
            ax.tick_params(axis='both', which='major', labelsize=f_s)
            ax.legend(fontsize=f_s)
            tensorboard_t = self.logger.experiment
            tensorboard_t.add_figure(
                f'test_datapd {i}', fig, self.current_epoch
            )
            plt.close(fig)
            ############################################
            fig, ax = plt.subplots(figsize=(13, 8))
            
            ax.step(t, p1[count].cpu().numpy(), label='first_fit', linewidth=l_w)
            ax.step(t, x1N[count].cpu().numpy(), label='first_data', linewidth=l_w)
            # ax.step(t, pf_chi2, label='first_Chi2', linewidth=l_w)

            ax.step(t, p2[count].cpu().numpy(), label='second_fit', linewidth=l_w)
            ax.step(t, x2N[count].cpu().numpy(), label='second_data', linewidth=l_w)
            # ax.step(t, ps_chi2, label='second_Chi2', linewidth=l_w)


            ax.set_xlabel('Time', fontsize=f_s)
            ax.set_ylabel('Voltage', fontsize=f_s)
            ax.tick_params(axis='both', which='major', labelsize=f_s)
            ax.legend(fontsize=f_s)
            tensorboard_t = self.logger.experiment
            tensorboard_t.add_figure(
                f'test_datap12 {i}', fig, self.current_epoch
            )
            plt.close(fig)

            ############################################     
            fig, ax = plt.subplots(figsize=(13, 8))
            # pdb.set_trace()
            ax.step(t, ps[count].cpu().numpy(), label='single_fit', linewidth=l_w)
            # ax.step(t, p_chi2_1, label='Chi2', linewidth=l_w)
            ax.step(t, x1[count].cpu().numpy(), label='single_data', linewidth=l_w)


            ax.set_xlabel('Time', fontsize=f_s)
            ax.set_ylabel('Voltage', fontsize=f_s)
            ax.tick_params(axis='both', which='major', labelsize=f_s)
            ax.legend(fontsize=f_s)
            tensorboard_t = self.logger.experiment
            tensorboard_t.add_figure(
                f'test_dataps {i}', fig, self.current_epoch
            )
            plt.close(fig)
            ############### saving ############################
            

            np.savetxt(f'{pulse_dir}/x1 {i}.txt', x1[count].cpu().detach().numpy())
            np.savetxt(f'{pulse_dir}/x2 {i}.txt', x2[count].cpu().detach().numpy())
            np.savetxt(f'{pulse_dir}/x1N {i}.txt', x1N[count].cpu().detach().numpy())
            np.savetxt(f'{pulse_dir}/x2N {i}.txt', x2N[count].cpu().detach().numpy())
            np.savetxt(f'{pulse_dir}/xd {i}.txt', xd[count].cpu().detach().numpy())
            np.savetxt(f'{pulse_dir}/ps {i}.txt', ps[count].cpu().detach().numpy())
            np.savetxt(f'{pulse_dir}/p1 {i}.txt', p1[count].cpu().detach().numpy())
            np.savetxt(f'{pulse_dir}/p2 {i}.txt', p2[count].cpu().detach().numpy())
            np.savetxt(f'{pulse_dir}/pd {i}.txt', pd[count].cpu().detach().numpy())
            np.savetxt(f'{pulse_dir}/dt {i}.txt', self.n_t_samples * dt[count].cpu().detach().numpy())

        np.savetxt(f'{pulse_dir}/probs_s.txt', probs_s.cpu().detach().numpy())
        np.savetxt(f'{pulse_dir}/probs_d.txt', probs_d.cpu().detach().numpy())
        # pdb.set_trace()
    def on_test_epoch_end(self):
        loss_ps = self.criterion['test_lossps'].compute()

        loss_p1 = self.criterion['test_lossp1'].compute()
        loss_p2 = self.criterion['test_lossp2'].compute()
        loss_pd = self.criterion['test_losspd'].compute()
        loss_c = self.criterion['test_lossc'].compute()
        loss_rd = self.criterion['test_lossrd'].compute()
        loss_t = self.criterion['test_losst'].compute()

        acc = self.acc_test.compute() * 100
        cm = self.cm_test.compute()
        cm_percentage = cm.float() / cm.sum(axis=1, keepdim=True) * 100
        ts, fd, fs, td = cm_percentage.ravel()

        snr_ps = self.snr_ps_test.compute()
        snr_pd = self.snr_pd_test.compute()
        snr_p1 = self.snr_p1_test.compute()
        snr_p2 = self.snr_p2_test.compute()


        self.log('test_loss_ps', loss_ps, on_epoch=True, prog_bar=False)

        self.log('test_loss_p1', loss_p1, on_epoch=True, prog_bar=False)
        self.log('test_loss_p2', loss_p2, on_epoch=True, prog_bar=False)
        self.log('test_loss_pd', loss_pd, on_epoch=True, prog_bar=True)
        self.log('test_loss_c', loss_c, on_epoch=True, prog_bar=False)
        self.log('test_loss_rd', loss_rd, on_epoch=True, prog_bar=False)
        self.log('test_loss_dt', loss_t, on_epoch=True, prog_bar=False)

        self.log('test_acc', acc, on_epoch=True, prog_bar=False)
        self.log('test_true_single', ts, on_epoch=True, prog_bar=False)
        self.log('test_false_double', fd, on_epoch=True, prog_bar=False)
        self.log('test_false_single', fs, on_epoch=True, prog_bar=False)
        self.log('test_true_double', td, on_epoch=True, prog_bar=False)

        self.log('test_snr_ps', snr_ps, on_epoch=True, prog_bar=False)
        self.log('test_snr_pd', snr_pd, on_epoch=True, prog_bar=False)
        self.log('test_snr_p1', snr_p1, on_epoch=True, prog_bar=False)
        self.log('test_snr_p2', snr_p2, on_epoch=True, prog_bar=False)

        self.criterion['test_lossps'].reset()
        self.criterion['test_lossp1'].reset()
        self.criterion['test_lossp2'].reset()
        self.criterion['test_losspd'].reset()
        self.criterion['test_lossc'].reset()
        self.criterion['test_lossrd'].reset()
        self.criterion['test_losst'].reset()

        self.cm_test.reset()
        self.acc_test.reset()
        self.snr_ps_test.reset()
        self.snr_pd_test.reset()
        self.snr_p1_test.reset()
        self.snr_p2_test.reset()

    ################################# Optimization #############################
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.opts.optim_method)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.opts.learning_rate, **self.opts.optim_options)
        
        if self.opts.scheduler_method=="CustomMultiGammaLR":
            # print('learning rate:', optimizer)
            scheduler_gen = schedulers_custom.CustomMultiGammaLR(
                optimizer,
                **self.opts.scheduler_options
            )
            return [optimizer], [scheduler_gen]
        elif self.opts.scheduler_method is not None:
            scheduler_gen = getattr(torch.optim.lr_scheduler, self.opts.scheduler_method)(
                optimizer, **self.opts.scheduler_options
            )
            return [optimizer], [scheduler_gen]
        else:
            return [optimizer]
        
    
    ################################ Other functions ########################################
    '''
    def logistic_function(self, t, t1, a1, k1, k2, ofs=0, k_scale=1):
        return a1 * torch.exp(-k_scale * k1 * (t - t1)) / (1 + torch.exp(-k_scale *k2 * (t - t1))) + ofs
    '''

    def logistic_function(self, t, ts, a, ks, rs):
        delta_k = rs - ks
        t0 = 9 / (delta_k + 1e-6 * (delta_k == 0).float())
        t0 = 0.15
        # Create a mask to avoid computing problematic values for t < t1 - 0.2
        mask = (t >= ts - t0).float()

        # Use a safe t value where mask is 0 to prevent inf/NaN
        safe_t = t * mask + (ts - t0) * (1 - mask)

        # Compute the logistic function only where mask is 1
        result = a * torch.exp(-ks * (safe_t - ts)) / \
                (1 + torch.exp(-rs * (safe_t - ts)))

        # Apply the mask to ensure the output is zero where needed
        result = result * mask
        
        return result
