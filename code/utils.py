
# coding: utf-8


import random
import torch
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def pred_eval(y_pred, y_test):
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    rmse = sqrt(mean_squared_error(y_pred, y_test))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
#     print('rmse: %.2f'%rmse)
#     print('mae: %.2f'%mae)
#     print('mape: %.2f'%mape)
    return rmse, mae, mape

