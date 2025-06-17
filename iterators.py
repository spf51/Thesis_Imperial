# Copyright 2021 Predicitve Intelligence Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as np
from jax import random, jit, vmap
from torch.utils import data
from functools import partial

class DeltaHelmholtzGenerator(data.Dataset):
    """
    Generator for Δ-PINNs Helmholtz on complex geometry.
    Yields a tuple of two batches:
      - (V_obs, t_obs), (phi_obs, _)
      - (V_elems, t_elems), (_ , _)
    """

    def __init__(self,
                 eigenfuncs,    # np.array (n_nodes, n_eigs)
                 phi_obs,       # np.array (n_times, n_nodes)
                 connectivity,  # np.array (n_elems, nodes_per_elem)
                 mesh_operator, # has .rows, .mass, .stiffness
                 times,         # np.array (n_times,)
                 mu_X=0.0,
                 sigma_X=1.0,
                 batch_size_data=64,
                 batch_size_res=64,
                 rng_key=random.PRNGKey(1234)):
        super().__init__()
        # store everything
        self.V          = np.array(eigenfuncs)
        self.phi_obs    = np.array(phi_obs)
        self.connectivity = np.array(connectivity)
        # FE weights per element × node
        self.mass       = np.array(mesh_operator.mass)
        self.stiffness  = np.array(mesh_operator.stiffness)
        self.times      = np.array(times)

        self.mu_X       = mu_X
        self.sigma_X    = sigma_X
        self.batch_size_data = batch_size_data
        self.batch_size_res  = batch_size_res

        self.key        = rng_key

    def __len__(self):
        # so torch knows how many to cycle through; you can pick any number
        return 10**6

    def __getitem__(self, _):
        # split rng and delegate
        self.key, subkey = random.split(self.key)
        return self.__data_generation(subkey)

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        # split off 5 subkeys
        k0, k1, k2, k3, k4 = random.split(key, 5)

        # --- DATA batch (node/time pairs) ---
        # sample times and nodes
        t_idx   = random.randint(k0, (self.batch_size_data,), 0, self.times.shape[0])
        n_idx   = random.randint(k1, (self.batch_size_data,), 0, self.V.shape[0])
        V_data  = self.V[n_idx]                           # (batch_data, n_eigs)
        t_data  = self.times[t_idx]                       # (batch_data,)
        phi_data= self.phi_obs[t_idx, n_idx]              # (batch_data,)

        # normalize inputs
        V_data = (V_data - self.mu_X)/self.sigma_X
        t_data = (t_data - self.mu_X)/self.sigma_X

        # dummy second target slot (unused)
        y_data = (phi_data, np.zeros_like(phi_data))

        # --- RESIDUAL batch (element/time pairs) ---
        e_idx   = random.randint(k2, (self.batch_size_res,), 0, self.connectivity.shape[0])
        t_idx2  = random.randint(k3, (self.batch_size_res,), 0, self.times.shape[0])

        # gather eigenfuncs at each element's nodes
        # shape → (batch_res, nodes_per_elem, n_eigs)
        gather = lambda ei: self.V[self.connectivity[ei]]
        V_elems = vmap(gather)(e_idx)
        t_res   = self.times[t_idx2]                      # (batch_res,)

        # normalize
        V_elems = (V_elems - self.mu_X)/self.sigma_X
        t_res    = (t_res - self.mu_X)/self.sigma_X

        # dummy targets for residual term
        dummy_res = np.zeros((self.batch_size_res,))      # unused

        # package as ((inputs),(targets)),((inputs),(targets))
        data_batch = ((V_data, t_data), y_data)
        res_batch  = ((V_elems, t_res), (dummy_res, dummy_res))

        return data_batch, res_batch

class Poisson1DGenerator(data.Dataset):
    def __init__(self, bcs_sampler, res_sampler,
                 mu_X = 0.0, sigma_X = 1.0,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.mu_X = mu_X
        self.sigma_X = sigma_X
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        X, y = self.__data_generation(subkey)
        return X, y

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        X_bc1, Y_bc1 = self.bcs_sampler[0].sample(self.batch_size//2, key)
        X_bc2, Y_bc2 = self.bcs_sampler[1].sample(self.batch_size//2, key)
        X_res, Y_res = self.res_sampler.sample(self.batch_size, key)
        # Normalize
        X_bc1 = (X_bc1 - self.mu_X)/self.sigma_X
        X_bc2 = (X_bc2 - self.mu_X)/self.sigma_X
        X_res = (X_res - self.mu_X)/self.sigma_X
        # Construct batch
        inputs  = (X_bc1, X_bc2, X_res)
        outputs = (Y_bc1, Y_bc2, Y_res)
        return inputs, outputs


class Wave1DGenerator(data.Dataset):
    def __init__(self, ics_sampler, bcs_sampler, res_sampler,
                 mu_X, sigma_X,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.mu_X = mu_X
        self.sigma_X = sigma_X
        self.ics_sampler = ics_sampler
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, targets = self.__data_generation(subkey)
        return inputs, targets

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        X_ic1, Y_ic1 = self.ics_sampler[0].sample(self.batch_size//3, key)
        X_bc1, Y_bc1 = self.bcs_sampler[0].sample(self.batch_size//3, key)
        X_bc2, Y_bc2 = self.bcs_sampler[1].sample(self.batch_size//3, key)
        X_ic2, Y_ic2 = self.ics_sampler[1].sample(self.batch_size, key)
        X_res, Y_res = self.res_sampler.sample(self.batch_size, key)
        # Normalize
        X_ic1 = (X_ic1 - self.mu_X)/self.sigma_X
        X_ic2 = (X_ic2 - self.mu_X)/self.sigma_X
        X_bc1 = (X_bc1 - self.mu_X)/self.sigma_X
        X_bc2 = (X_bc2 - self.mu_X)/self.sigma_X
        X_res = (X_res - self.mu_X)/self.sigma_X
        # Make inputs, outputs
        inputs  = (X_ic1, X_ic2, X_bc1, X_bc2, X_res)
        outputs = (Y_ic1, Y_ic2, Y_bc1, Y_bc2, Y_res)
        return inputs, outputs


class IncNavierStokes4DFlowMRIGenerator(data.Dataset):
    def __init__(self, dat_sampler, res_sampler,
                 mu_X = 0.0, sigma_X = 1.0,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.mu_X = mu_X
        self.sigma_X = sigma_X
        self.dat_sampler = dat_sampler
        self.res_sampler = res_sampler
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, targets = self.__data_generation(subkey)
        return inputs, targets

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        X_dat, Y_dat = self.dat_sampler.sample(self.batch_size, key)
        X_res, Y_res = self.res_sampler.sample(self.batch_size, key)
        # Normalize
        X_dat = (X_dat - self.mu_X)/self.sigma_X
        X_res = (X_res - self.mu_X)/self.sigma_X
        # Make inputs, outputs
        inputs  = (X_dat, X_res)
        outputs = (Y_dat, Y_res)
        return inputs, outputs

class AllenCahn1DGenerator(data.Dataset):
    def __init__(self, ics_sampler, bcs_sampler,
                 mu_X = 0.0, sigma_X = 1.0,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.mu_X = mu_X
        self.sigma_X = sigma_X
        self.ics_sampler = ics_sampler
        self.bcs_sampler = bcs_sampler
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, targets = self.__data_generation(subkey)
        return inputs, targets

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        X0, Y0 = self.ics_sampler.sample(self.batch_size, key)
        X1_bc1, Y1_bc1  = self.bcs_sampler[0].sample(self.batch_size//2, key)
        X1_bc2, Y1_bc2  = self.bcs_sampler[1].sample(self.batch_size//2, key)
        # Normalize
        X0 = (X0 - self.mu_X)/self.sigma_X
        X1_bc1 = (X1_bc1 - self.mu_X)/self.sigma_X
        X1_bc2 = (X1_bc2 - self.mu_X)/self.sigma_X
        # Make inputs, outputs
        inputs  = (X0, X1_bc1, X1_bc2)
        outputs = (Y0, Y1_bc1, Y1_bc2)
        return inputs, outputs


class Beltrami3DGenerator(data.Dataset):
    def __init__(self, ics_sampler, bcs_sampler, res_sampler,
                 mu_X = 0.0, sigma_X = 1.0,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.mu_X = mu_X
        self.sigma_X = sigma_X
        self.ics_sampler = ics_sampler
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, targets = self.__data_generation(subkey)
        return inputs, targets

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        X_ics, Y_ics = self.ics_sampler.sample(self.batch_size//7, key)
        X_bc1, Y_bc1 = self.bcs_sampler[0].sample(self.batch_size//7, key)
        X_bc2, Y_bc2 = self.bcs_sampler[1].sample(self.batch_size//7, key)
        X_bc3, Y_bc3 = self.bcs_sampler[2].sample(self.batch_size//7, key)
        X_bc4, Y_bc4 = self.bcs_sampler[3].sample(self.batch_size//7, key)
        X_bc5, Y_bc5 = self.bcs_sampler[4].sample(self.batch_size//7, key)
        X_bc6, Y_bc6 = self.bcs_sampler[5].sample(self.batch_size//7, key)
        X_res, Y_res = self.res_sampler.sample(self.batch_size, key)
        # Normalize
        X_ics = (X_ics - self.mu_X)/self.sigma_X
        X_bc1 = (X_bc1 - self.mu_X)/self.sigma_X
        X_bc2 = (X_bc2 - self.mu_X)/self.sigma_X
        X_bc3 = (X_bc3 - self.mu_X)/self.sigma_X
        X_bc4 = (X_bc4 - self.mu_X)/self.sigma_X
        X_bc5 = (X_bc5 - self.mu_X)/self.sigma_X
        X_bc6 = (X_bc6 - self.mu_X)/self.sigma_X
        X_res = (X_res - self.mu_X)/self.sigma_X
        # Make inputs, outputs
        inputs  = (X_ics, X_bc1, X_bc2, X_bc3, X_bc4, X_bc5, X_bc6, X_res)
        outputs = (Y_ics, Y_bc1, Y_bc2, Y_bc3, Y_bc4, Y_bc5, Y_bc6, Y_res)
        return inputs, outputs
