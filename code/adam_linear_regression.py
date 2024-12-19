import jax.numpy as jnp
import numpy as np
from jax import random, jit, grad, hessian, vmap
import optax

import itertools
from functools import partial
from tqdm import trange
import pandas as pd
import matplotlib.pyplot as plt
import sys

feature_file = sys.argv[1]
data_file = sys.argv[2]

feature_string = feature_file.split('/')[-1].split('.')[0]
data_string = data_file.split('/')[-1].split('.')[0]

class Regressor:
    def __init__(self, num_features, rng_key):
        self.rng_key = rng_key
        self.params = random.normal(rng_key, (num_features, 1))

        # Optimizer
        lr = optax.exponential_decay(1e-3, transition_steps=5000, decay_rate=0.9)
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []

    def apply(self, params, X):
        return jnp.dot(X, params)

    def loss(self, params, batch):
        X, y = batch
        y_fit = self.apply(params, X)
        y_fit = y_fit.squeeze()
        return jnp.mean((y - y_fit)**2)

    @partial(jit, static_argnums=(0,))
    def step(self, params, opt_state, batch):
        grads = grad(self.loss)(params, batch)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    
    def train(self, batch, iterations = 30000):
        pbar = trange(iterations)
        for it in pbar:
            self.params, self.opt_state = self.step(self.params, self.opt_state, batch)
            if it % 10 == 0:
                loss = self.loss(self.params, batch)
                self.loss_log.append(loss)
                pbar.set_postfix({'loss': loss})

def DataReader(features, dataset, train_size=0.8):
    with open(features, 'r') as featfile:
        feats = featfile.read().split(',')[:-1]

    data = pd.read_csv(dataset)

    y = jnp.array(data['E_ads'])

    to_drop = ['Unnamed: 0','metadata', 'chem','site','E_ads']
    to_drop = [col for col in to_drop if col in data.columns]
    metadata = data[to_drop] if to_drop else pd.DataFrame()

    x_full = data.drop(to_drop, axis=1)
    x = jnp.array(x_full[feats])

    xmean = jnp.mean(x, axis=0)
    xstd = jnp.std(x, axis=0)
    xstd = jnp.where(xstd == 0, 1, xstd)

    x = (x - xmean) / xstd

    num_samples = x.shape[0]
    indices = np.random.permutation(num_samples)
    train_size = int(train_size * num_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return x_train, y_train, x_test, y_test, metadata


x_train, y_train, x_test, y_test, _ = DataReader(feature_file, data_file)

num_features = x_train.shape[1]

model = Regressor(num_features, random.PRNGKey(0))
model.train((x_train, y_train), iterations=20000)

dfloss = pd.DataFrame()
dfloss['Loss'] = model.loss_log

dfloss.to_csv(f'/home/intriago/scripts/submissions/ML/enm5310/nolr_results/NOLR_loss_{data_string}_{feature_string}.csv')

y_train_pred = model.apply(model.params, x_train)
y_test_pred = model.apply(model.params, x_test)

dftrain = pd.DataFrame()
dftrain['True'] = y_train
dftrain['Predicted'] = y_train_pred

dftrain.to_csv(f'/home/intriago/scripts/submissions/ML/enm5310/nolr_results/NOLR_train_data_{data_string}_{feature_string}.csv')

dftest = pd.DataFrame()
dftest['True'] = y_test
dftest['Predicted'] = y_test_pred

dftest.to_csv(f'/home/intriago/scripts/submissions/ML/enm5310/nolr_results/NOLR_test_data_{data_string}_{feature_string}.csv')