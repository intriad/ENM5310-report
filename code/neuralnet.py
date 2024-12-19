import jax.numpy as jnp
import numpy as np
from jax import random, vmap, grad, jit

import jax
import flax.linen as nn
import optax
from sklearn.metrics import mean_absolute_error
import pandas as pd
from torch.utils import data

import itertools
from functools import partial
from tqdm import trange
import matplotlib.pyplot as plt

import sys
import pandas as pd

feature_file = sys.argv[1]
data_file = sys.argv[2]

feature_string = feature_file.split('/')[-1].split('.')[0]
data_string = data_file.split('/')[-1].split('.')[0]

class MLP(nn.Module):
    layer_dim: int
    num_layers: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers):
            x = nn.Dense(self.layer_dim)(x)
            x = nn.tanh(x)
        out = nn.Dense(self.out_dim)(x)

        return out

class NN:
    def __init__(self, model, num_features, rng_key = random.PRNGKey(0)):
        self.model = model
        x = jax.random.uniform(rng_key, (1,num_features), minval=-2, maxval=2)
        self.params = model.init(rng_key, x)

        # Optimizer
        lr = optax.exponential_decay(1e-4, transition_steps=1000, decay_rate=0.9)
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []

    def loss(self, params, batch):
        x, y = batch
        y_fit = self.model.apply(params, x)
        y_fit = y_fit.squeeze()
        MSE = jnp.mean((y_fit - y) ** 2)
        l2_norm = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        return MSE + 0.01*l2_norm

    @partial(jit, static_argnums=(0,))
    def step(self, params, opt_state, batch):
        grads = grad(self.loss)(params, batch)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    
    def train(self, dataset, nIter = 10000):
        data = iter(dataset)
        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            batch = next(data)
            self.params, self.opt_state = self.step(self.params, self.opt_state, batch)
            # Logger
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

class DataGenerator(data.Dataset):
    def __init__(self, data, target, 
                 batch_size=128, 
                 rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.data = data
        self.target = target
        self.N = target.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __data_generation(self, key, data, target):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        return data[idx], target[idx]
    
    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        data, target = self.__data_generation(self.key, self.data, self.target)
        return data, target

x_train, y_train, x_test, y_test, _ = DataReader(feature_file, data_file)

dataset = DataGenerator(x_train, y_train, batch_size=128)

num_features = x_train.shape[1]

model = NN(MLP(layer_dim=256, num_layers=2, out_dim=1), num_features)
model.train(dataset, nIter=10000)

y_train_pred = vmap(model.model.apply, in_axes=(None, 0))(model.params, x_train)
y_test_pred = vmap(model.model.apply, in_axes=(None, 0))(model.params, x_test)

MAE_train = mean_absolute_error(y_train, y_train_pred)
MAE_test = mean_absolute_error(y_test, y_test_pred)

plt.plot(model.loss_log)
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.savefig(f'/home/intriago/scripts/submissions/ML/enm5310/figures/MLP_loss_{data_string}_{feature_string}.png')

df_loss = pd.DataFrame()
df_loss['Loss'] = model.loss_log

df_loss.to_csv(f'/home/intriago/scripts/submissions/ML/enm5310/mlp_results/MLP_loss_{data_string}_{feature_string}.csv')

y_train_min = jnp.min(y_train)
y_train_max = jnp.max(y_train)

y_test_min = jnp.min(y_test)
y_test_max = jnp.max(y_test)

plt.figure(figsize=(5, 5))
plt.plot(y_train, y_train, 'k', linewidth=0.5)
plt.plot(y_train, y_train_pred, 'bo', markersize=2.5)
plt.xlim(y_train_min, y_train_max)
plt.ylim(y_train_min, y_train_max)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title(f'Train (MAE = {MAE_train})')

plt.savefig(f'/home/intriago/scripts/submissions/ML/enm5310/figures/MLP_train_data_{data_string}_{feature_string}.png')

y_test_min = jnp.min(y_test)
y_test_max = jnp.max(y_test)

plt.figure(figsize=(5, 5))
plt.plot(y_test, y_test, 'k', linewidth=0.5)
plt.plot(y_test, y_test_pred, 'bo', markersize=2.5)
plt.xlim(y_test_min, y_test_max)
plt.ylim(y_test_min, y_test_max)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title(f'Test (MAE = {MAE_test})')

plt.savefig(f'/home/intriago/scripts/submissions/ML/enm5310/figures/MLP_test_data_{data_string}_{feature_string}.png')

df_train = pd.DataFrame()
df_train['True'] = y_train
df_train['Predicted'] = y_train_pred

df_train.to_csv(f'/home/intriago/scripts/submissions/ML/enm5310/mlp_results/MLP_train_data_{data_string}_{feature_string}.csv')

df_test = pd.DataFrame()
df_test['True'] = y_test
df_test['Predicted'] = y_test_pred

df_test.to_csv(f'/home/intriago/scripts/submissions/ML/enm5310/mlp_results/MLP_test_data_{data_string}_{feature_string}.csv')