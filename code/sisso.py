import sys
import numpy as np
import pandas as pd
import argparse
import time
import sissopp
from sissopp.sklearn import SISSORegressor
from sissopp.postprocess.load_models import load_model
import os
import sys
from copy import deepcopy
import heapq
from scipy import stats
import random
import seaborn as sb # import seaborn as sns
import statsmodels.api as sm
import sklearn
from sklearn.feature_selection import RFE
from sklearn import linear_model
from sklearn import svm
import sklearn.metrics
from sklearn.metrics import r2_score
from sklearn import decomposition
from sklearn.ensemble import ExtraTreesRegressor

np.random.seed(42)

parser = argparse.ArgumentParser(description="SISSO Settings")

parser.add_argument(
"--n",
default=1,
type=int,
help="dimension for the descriptor (default: 1)",
)

parser.add_argument(
"--r",
default=0,
type=int,
help="rung (default: 0)",
)

parser.add_argument(
"--feature_list_file",
default='features.txt',
type=str,
help='path to .txt file with list of features you want to run (default: features.txt)',
)

parser.add_argument(
"--feature_set_file",
default='all_features.csv',
type=str,
help='path to .csv file with list of all features you want to run (default: all_features.csv)',
)

parser.add_argument(
"--n_features",
default=20,
type=int,
help='number of features to select (default: 10)',
)

args = parser.parse_args(sys.argv[1:])

n = args.n
r = args.r

feature_txt = open(args.feature_list_file, "r")
content = feature_txt.read()
features = content.split(",")[:-1]
feature_txt.close()

def random_cols(X):
    r = np.asarray(range(0,len(list(X.columns))))
    random.shuffle(r)
    X = X.iloc[:, r] # randomize the columns
    return X

def fix_cols(cols):
    for c in range(len(cols)):
        if '-' in cols[c]:
            cols[c] = cols[c].replace('-','_')
    return cols

def get_fit_from_dat(file_path):
    y_train_meas = []
    y_train_fit = []
    with open(file_path) as f:
        for line in f:
            if line.startswith('sample'):
                temp = line.split(',')
                y_train_meas.append(float(temp[1]))
                y_train_fit.append(float(temp[2]))

    return np.array(y_train_meas), np.array(y_train_fit)
        

#normalization of the data
def data_norm(X,y):
    #X is 2D np.array
    #y is 1D np.array
    avgs = []
    stddevs = []
    X_0 = X.copy()
    y_0 = y.copy()
    for k in range(X.shape[1]):
        avg = np.mean(X[:,k])
        temp = np.std(X[:,k])
        if temp == 0:
            temp = 1 # avoid dividing by 0, could also set to a really small number
        avgs.append(avg)
        stddevs.append(temp)
        X[:,k] = (X[:,k] - avg)/temp

    y_avg = np.mean(y)
    y_std = np.std(y)
    #     print('y_avg:',y_avg)
    #     print('y_std:',y_std)
    if y_std == 0:
        y_std = 1 # avoid dividing by 0, could also set to a really small number
    y_norm = (y - y_avg)/y_std
    #     print('--------------------')

    return X, y_norm, X_0, y_0, y_avg, y_std, avgs, stddevs


# split into testing and training sets
#def train_test_split(train_split,[args]):
def train_test_split(X,y,train_split):
    #X is 2D np.array
    #y is 1D np.array
    #train_split is the fraction of data that will be in the TRAINING set 
    # (i.e. 0.75 for a 75-25 training-testing split)
    #assume all data been randomized already
    num_train = round(X.shape[0]*train_split)
    X_train = X[:num_train,:]
    y_train = y[:num_train]
    X_test = X[num_train:,:]
    y_test = y[num_train:]

    X_train, y_train, X_train_0, y_train_0, y_train_avg, y_train_std, X_train_avgs, X_train_std = data_norm(X_train,y_train)
    X_test, y_test, X_test_0, y_test_0, y_test_avg, y_test_std, X_test_avgs, X_test_std = data_norm(X_test,y_test)

    return [X_train, y_train, X_train_0, y_train_0, y_train_avg, y_train_std, X_train_avgs, X_train_std, X_test, y_test, X_test_0, y_test_0, y_test_avg, y_test_std, X_test_avgs, X_test_std]


# Define the data set
file_path =  args.feature_set_file
#you'll need to change this for each featureset
all_features = pd.read_csv(file_path,delimiter=',')

X = all_features.copy()
# Randomize the rows while keeping data together
X = X.sample(frac = 1).reset_index(drop=True) # shuffle/randomize the rows
#now we get y in the same order that all the rows have been shuffled
y = X['E_ads'].to_numpy()
to_drop = ['Unnamed: 0','metadata', 'chem','site','E_ads']
to_drop = [col for col in to_drop if col in X.columns]
keep_meta = X[to_drop] if to_drop else pd.DataFrame() # keep this data somewhere in the right order
#remove columns from X we don't need for now
X = X.drop(to_drop,axis=1)

X_new = {}
for f in features:
	temp = {f: X[f]}
	X_new = {**X_new,**temp}

X_run = pd.DataFrame(X_new)

X_run = random_cols(X_run) #randomize the columns
cols = fix_cols(list(X_run.columns)) #keep a list of the columns in the right order

X_run_np = X_run.to_numpy()

train_split = 0.90

#[X_train, y_train, X_train_0, y_train_0, y_train_avg, y_train_std, X_train_avgs, X_train_std,

[X_run_train, y_run_train, X_run_train_0, y_run_train_0, y_run_train_avg, y_run_train_std, X_run_train_avgs, X_run_train_std, X_run_test, y_run_test,  X_run_test_0, y_run_test_0, y_run_test_avg, y_run_test_std, X_run_test_avgs, X_run_test_std] = train_test_split(X_run_np,y,train_split)

#save X_avgs and X_stdevs for unsscaling later
X_avgs = pd.DataFrame(np.asarray(X_run_train_avgs).reshape((1,-1)),columns=cols)
X_stds = pd.DataFrame(np.asarray(X_run_train_std).reshape((1,-1)),columns=cols)

X_avgs.to_csv('X_avgs.csv')
X_stds.to_csv('X_stds.csv')

s = args.feature_list_file.split('/')[-1][:-4]

start_time = time.time()
# Define the regressor and fit the data
sisso_reg = SISSORegressor(
    prop_label = 'E_ads',
    prop_unit = 'eV',
    max_rung=r,
    allowed_ops = ["add", "sub", "mult", "div", "sq", "cb", "inv", "sqrt"],
    n_dim = n,
    n_sis_select = args.n_features,
    n_residual = args.n_features,
)

sisso_reg.workdir = 'SISSO_run_' + s +'_'+'R'+str(r) +'_'+ str(n) + 'D'

#X and y data is zero mean with unit var=1
sisso_reg.fit(X_run_train, y_run_train, columns=cols)

model_path = 'SISSO_run_' + s +'_'+ 'R' + str(r) + '_' + str(n) + 'D' + '/models/train_dim_' + str(n) + '_model_0.dat'

sisso_model = load_model(model_path)

t = time.time() - start_time
print("--- %s seconds ---" % round((time.time() - start_time),3))

# get the prediction with the X_test and y_test
X_test_pd = pd.DataFrame(X_run_test,columns=cols)
y_test_fit = sisso_reg.predict(X_test_pd)

#Print model specs
#calculate test MAE
test_mae = sklearn.metrics.mean_absolute_error(y_run_test, y_test_fit)
#calculate train MAE

y_train_meas, y_train_fit = get_fit_from_dat(model_path)

train_mae = sklearn.metrics.mean_absolute_error(y_train_meas, y_train_fit)

df_test = pd.DataFrame()
df_test['True'] = y_run_test
df_test['Predicted'] = y_test_fit

model_path_split = model_path.split('train_dim')[0]

df_test.to_csv(f'{model_path_split}test_data.csv')

df_train = pd.DataFrame()
df_train['True'] = y_train_meas
df_train['Predicted'] = y_train_fit

df_train.to_csv(f'{model_path_split}train_data.csv')

#calculate total MAE
y_fit_total = np.concatenate((y_test_fit.reshape((-1)), y_train_fit))
y_meas_total = np.concatenate((y_run_test, y_train_meas))
total_mae = sklearn.metrics.mean_absolute_error(y_meas_total, y_fit_total)
abs_err = abs(y_meas_total - y_fit_total)
maxae = max(abs_err)
r_squared = sklearn.metrics.r2_score(y_meas_total, y_fit_total)
RMSE_train = np.sqrt((sklearn.metrics.mean_squared_error(y_train_meas, y_train_fit)))
RMSE_test = np.sqrt(sklearn.metrics.mean_squared_error(y_run_test, y_test_fit))
RMSE_total = np.sqrt(sklearn.metrics.mean_squared_error(y_meas_total, y_fit_total))

print('coefs: ', sisso_model.coefs)
print('RMSE_train: ', RMSE_train)
print('RMSE_test: ', RMSE_test)
print('RMSE_total: ', RMSE_total)
print('MaxAE: ', maxae)
print('MAE_train: ', train_mae)
print('MAE_test: ', test_mae)
print('MAE_total: ', total_mae)
print('R^2: ', r_squared)
print('Use these to unnormalize the data for the plots:' )
print('y_train_avg: ', y_run_train_avg)
print('y_train_std: ', y_run_train_std)
print('y_test_avg: ', y_run_test_avg)
print('y_test_std: ', y_run_test_std)


