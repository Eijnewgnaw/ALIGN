import os
import random
import sys
import numpy as np
import scipy.io as sio
from scipy import sparse
from sklearn.model_selection import train_test_split
from utils import util


def load_multiview_data(config):
    data_name = config['dataset']
    main_dir = sys.path[0]
    X_list = []
    Y_list = []

    if data_name in ['Scene_15']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'Scene_15.mat'))
        X = mat['X'][0]
        X_list.append(X[0].astype('float32'))  # 20
        X_list.append(X[1].astype('float32'))  # 59
        X_list.append(X[2].astype('float32'))  # 40
        Y_list.append(np.squeeze(mat['Y']))
        Y_list.append(np.squeeze(mat['Y']))

    elif data_name in ['LandUse_21']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'LandUse_21.mat'))
        train_x = []
        train_x.append(sparse.csr_matrix(mat['X'][0, 0]).A)  # 20
        train_x.append(sparse.csr_matrix(mat['X'][0, 1]).A)  # 59
        train_x.append(sparse.csr_matrix(mat['X'][0, 2]).A)  # 40
        index = random.sample(range(train_x[0].shape[0]), 2100)
        for view in [1, 2, 0]:
            x = train_x[view][index]
            y = np.squeeze(mat['Y']).astype('int')[index]
            X_list.append(x)
            Y_list.append(y)

    elif data_name in ['Caltech101-20']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        X = mat['X'][0]
        for view in [3, 4, 5]:
            x = X[view]
            x = util.normalize(x).astype('float32')
            y = np.squeeze(mat['Y']).astype('int')
            X_list.append(x)
            Y_list.append(y)

    elif data_name in ['20newsgroups']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', '20newsgroups.mat'))
        X = mat['data'][0]
        X_list.append(X[0].T.astype('float32'))
        X_list.append(X[1].T.astype('float32'))
        X_list.append(X[2].T.astype('float32'))
        Y_list.append(np.squeeze(mat['truelabel'][0][0]))
        Y_list.append(np.squeeze(mat['truelabel'][0][0]))

    return X_list, Y_list