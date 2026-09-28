# net.py
import pdb

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
# import control.sean.utils as ut

__all__ = ['SAEFit']


class Identity(nn.Module):

    def __init__(self, ndim, nout):
        super(Identity, self).__init__()
        self.identity = nn.Linear(ndim, nout)

    def forward(self, x):
        z = self.identity(x)
        return z


    def __init__(self, ndim, nout, r, hdl_enc, hdl_tgt, temperature):
        super(VAE, self).__init__()
        self.temperature = temperature
        self.encoder = nn.Sequential(
            nn.Linear(ndim, 2 * hdl_enc),
            nn.ReLU(),
            nn.Linear(2 * hdl_enc, 2 * hdl_enc),
            nn.ReLU(),
            nn.Linear(2 * hdl_enc, 2 * hdl_enc),
            nn.PReLU(),
            nn.Linear(2 * hdl_enc, hdl_enc),
            nn.ReLU(),
            nn.Linear(hdl_enc, r * 2)  # *2 for mean and variance

        )
        self.decoder = nn.Sequential(
            nn.Linear(r, hdl_tgt),
            nn.ReLU(),
            nn.Linear(hdl_tgt, 2 * hdl_tgt),
            nn.ReLU(),
            nn.Linear(2 * hdl_tgt, 2 * hdl_tgt),
            nn.ReLU(),
            nn.Linear(2 * hdl_tgt, 2 * hdl_tgt),
            nn.PReLU(),
            nn.Linear(2 * hdl_tgt, nout),
            # nn.Sigmoid()  # Output between 0 and 1
        )
        self.latent_dim = r
        self.input_dim = ndim

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)  # Split into mean and log variance
        return mu, logvar

    def reparameterize(self, mu, logvar):
        var = torch.exp(0.5 * logvar)
        eps = torch.randn_like(logvar)
        eps_gen = torch.randn_like(logvar)
        z = mu + eps * var
        z_gen = mu + eps_gen * var + torch.randn_like(z) * self.temperature
        # z_gen = mu + eps * logvar
        # pdb.set_trace()
        return z, z_gen

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z, z_gen = self.reparameterize(mu, logvar)
        return z, self.decode(z), self.decode(z_gen), mu, logvar
    


class SAEFit(nn.Module):

    def __init__(self, ndim, nout, r, hdl_enc, hdl_tgt, hdl_est, n_param, t1_m=0.4,t2_m=0.5):
        super(SAEFit, self).__init__()
        self.t1_m = t1_m
        self.t2_m = t2_m

        self.t1_lim = [0.0, 1.0]
        self.a1s_lim = [0.0, 10.0]
        self.a1_lim = [0.0, 10.0]
        self.k1_lim = [0.0, 20.0]
        self.r1_lim = [0.0, 200.0]

        self.t2_lim = [0.000, 1.0]
        self.a2_lim = [0.0, 10.0]
        self.k2_lim = [0.0, 20.0]
        self.r2_lim = [0.0, 200.0]

        self.bounds = [self.t1_lim, self.a1s_lim, self.a1_lim ,self.k1_lim, self.r1_lim,
                       self.t2_lim,  self.a2_lim, self.k2_lim, self.r2_lim]


        self.encoder = nn.Sequential(
            nn.Linear(ndim, 2 * hdl_enc),
            nn.ReLU(),
            nn.Linear(2 * hdl_enc, 2 * hdl_enc),
            nn.ReLU(),
            nn.Linear(2 * hdl_enc, 2 * hdl_enc),
            nn.ReLU(),
            # nn.Linear(2 * hdl_enc, 1 * hdl_enc),
            # nn.ReLU(),
            nn.Linear(2 * hdl_enc, r),
            # nn.ReLU(),
        )
        # self.encoder = nn.LSTM(ndim, r, batch_first=True)

        self.decoder = nn.Sequential(
            nn.Linear(r, 1 * hdl_tgt),
            nn.ReLU(),
            nn.Linear(1 * hdl_tgt, 2 * hdl_tgt),
            nn.ReLU(),
            # nn.Linear(2 * hdl_tgt, 2 * hdl_tgt),
            # nn.ReLU(),
            nn.Linear(2 * hdl_tgt, nout),
        )

        self.estimator = nn.Sequential(
            nn.Linear(r, 2 * hdl_est),
            nn.ReLU(),
            # nn.Linear(2 * hdl_est, 2 * hdl_est),
            # nn.ReLU(),
            nn.Linear(2 * hdl_est, 1 * hdl_est),
            nn.ReLU(),
            nn.Linear(1 * hdl_est, n_param),

        )

        self.classifier = nn.Sequential(
            nn.Linear(r, 2 * hdl_est),
            nn.ReLU(),
            # nn.Linear(2 * hdl_est, hdl_est),
            # nn.ReLU(),
            nn.Linear(2 * hdl_est, 2)
        )

        self.base = nn.Parameter(torch.tensor([self.t1_m, 0.5, 0.5, 0.5, 0.5, self.t2_m, 0.5, 0.5, 0.5]).unsqueeze(1))
                                                # t1      a1s   a1  k1   r1       t2      a2   k2   r2
    def boundary_conditions(self, bases, boundries):
        outs = []
        for i in range(len(boundries)):
            if bases.shape[1] == 1:
                outs.append(torch.clamp(bases[i], min=boundries[i][0], max=boundries[i][1]))
            else:
                outs.append(torch.clamp(bases[:,i], min=boundries[i][0], max=boundries[i][1]).unsqueeze(1))
                
        return outs
    
    def forward(self, x, control=False):

        z = self.encoder(x)
        out = self.decoder(z)
        var = self.estimator(z)
        clas = self.classifier(z)

        
        if control:
            [t1, a1s, a1, k1, r1, t2, a2, k2, r2] = self.boundary_conditions(self.base, self.bounds)

        else:
            [t1, a1s, a1, k1, r1, t2, a2, k2, r2] = self.boundary_conditions(var + self.base.T, self.bounds)


        # [t1, a1s, a1, k1, k2, ofs1, t2, a2, k3, k4, ofs2] = self.boundary_conditions(var, self.bounds)

        return out, clas,\
        t1, a1s, a1, k1, r1, t2, a2, k2, r2
