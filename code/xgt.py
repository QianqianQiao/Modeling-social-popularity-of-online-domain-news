
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pickle
import os
import argparse
from utils import pred_eval
from base_models import xgt_regressor
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument("--gather_point", "-p", help="gather point, in hours", type=int, default = 5)
parser.add_argument("--data_type", "-d", help="type of data: time, event, news, all", type=str, default = 'time')
parser.add_argument("--objective", "-obj", help="objective, reg:gamma, reg:linear", type=str, default = 'reg:gamma')
args = parser.parse_args()


# parameters
GATHER_POINT = args.gather_point # in hours
INTERVAL = 0.25 # in hours

lr = [0.05,0.1,0.15, 0.2]
max_d = [4,8,10,15]
estimators = [50, 100]
'''
lr = [0.15]
max_d = [5]
estimators = [50]
'''
obj = args.objective

data_type = args.data_type
name = data_type+'_' + str(GATHER_POINT)
path_data = "../data/"
path_log  = "../results/xgboost/"+name+"_log.txt"
path_result = "../results/xgboost/"
path_models = "../results/xgboost/models/"


# read data
train_df = pd.read_pickle(path_data + 'train.p')
test_df = pd.read_pickle(path_data + 'test.p')
train_news = pd.read_pickle(path_data+'train_news.p')
test_news = pd.read_pickle(path_data+'test_news.p')

train_df.drop(index = 12797, inplace=True)
train_news.drop(index = 12797, inplace = True)
# data organization for xgboost model
if data_type == 'time':
    X_train = train_df.iloc[:, list(range(1, int(GATHER_POINT / INTERVAL)+2))].values
    X_test = test_df.iloc[:, list(range(1, int(GATHER_POINT / INTERVAL)+2))].values
elif data_type == 'event':
    train_df['vector_sum'] = train_df.apply(lambda x: sum(x['tw_vector'][:x['tweet_amount_'+str(GATHER_POINT)+'.0h']]), axis=1)
    test_df['vector_sum'] = test_df.apply(lambda x: sum(x['tw_vector'][:x['tweet_amount_'+str(GATHER_POINT)+'.0h']]), axis=1)
    X_train = np.array(list(train_df['vector_sum']))
    X_test =np.array(list(test_df['vector_sum']))
elif data_type == 'news':
    X_train = list(train_news['vector'])
    X_test = list(test_news['vector'])
elif data_type == 'all':
    train_df['vector_sum'] = train_df.apply(lambda x: sum(x['tw_vector'][:x['tweet_amount_'+str(GATHER_POINT)+'.0h']]), axis=1)
    test_df['vector_sum'] = test_df.apply(lambda x: sum(x['tw_vector'][:x['tweet_amount_'+str(GATHER_POINT)+'.0h']]), axis=1)
    X_train = np.concatenate((train_df.iloc[:, list(range(1, int(GATHER_POINT / INTERVAL)+2))].values, 
                              np.array(list(train_df['vector_sum'])),
                              np.array(list(train_news['vector']))), axis = 1)
    X_test = np.concatenate((test_df.iloc[:, list(range(1, int(GATHER_POINT / INTERVAL)+2))].values, 
                          np.array(list(test_df['vector_sum'])),
                          np.array(list(test_news['vector']))), axis = 1)
    
y_train = list(train_df['label'])
y_test = list(test_df['label'])


# run model
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
model = xgt_regressor(lr, max_d, estimators, X_tr, X_val, y_tr, y_val, obj)
# inference
pred = model[0].predict(X_test)
rmse, mae, mape = pred_eval(pred, y_test)

# save results
np.save(path_result+name+"_output.npy", pred)
pickle.dump(model[0], open(path_models+name+".pth", "wb"))

with open(path_log, "a") as text_file:
    text_file.write("\n\n ------ Parameters and Evaluation Results ------- \n")
    text_file.write("objective type : %s \n"%(obj))
    text_file.write("learning rate : %.4f \n"%(model[1][0]))
    text_file.write("max depth : %.4f \n"%(model[1][1]))
    text_file.write("number of estimators : %.4f \n\n"%(model[1][2]))

    text_file.write("best rmse : %.4f \n"%(rmse))
    text_file.write("best mae : %.4f \n"%(mae))
    text_file.write("best mape : %.4f \n"%(mape))

