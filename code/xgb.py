
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import pickle as pickle
import os
from sklearn.model_selection import train_test_split 
import time
import argparse
from utils import pred_eval
from base_models import xgt_regressor


# In[ ]:


# read data
# parameters
parser = argparse.ArgumentParser()
parser.add_argument("--gather_point", "-p", help="gather point, in hours", type=int, default = 5)
args = parser.parse_args()

GATHER_POINT = args.gather_point # in hours
INTERVAL = 0.25 # in hours

lr = [0.05]
max_d = [4]
estimators = [10]

name = str(GATHER_POINT)
path_data = "../data/"
path_log  = "../results/xgboost_e/"+name+"_log.txt"
path_result = "../results/xgboost_e/"
#path_models = "../results/xgboost_e/models/"

train_df = pd.read_pickle(path_data + 'train.p')
test_df = pd.read_pickle(path_data + 'test.p')
train_news = pd.read_pickle(path_data+'train_news.p')
test_news = pd.read_pickle(path_data+'test_news.p')

pred_event_test = np.load('../results/neuralpp/'+name+'_output.npy')
pred_event_train = np.load('../results/neuralpp/'+name+'_train_output.npy')
pred_event_test = pred_event_test[:,:2]
pred_event_train = pred_event_train[:,:2]
# In[25]:


# form data
train_df['tweet_amount_5.0h'] = train_df['tw_vector'].map(len)
train_df['delta_t'] = train_df.apply(lambda x: [(x['time_list'][i]-x['time_list'][i-1])/24
                                                for i in range(1,x['tweet_amount_'+str(GATHER_POINT)+'.0h'])],
                                     axis=1)
train_df['len'] = train_df['delta_t'].map(len)
train_df = train_df[train_df['len']>0]
train_df = train_df.reset_index(drop=True)
train_news = train_df[['url','label']].merge(train_news[['url','vector']], on='url', how='left')

X_train = train_df.iloc[:, list(range(1, int(GATHER_POINT / INTERVAL)+2))].values
X_train2 = list(train_news['vector'])
y_train = list(train_df['label'])
dy_train = np.array(y_train).reshape((-1,1))-np.array(pred_event_train)

test_df['tweet_amount_5.0h'] = test_df['tw_vector'].map(len)
test_df['delta_t'] = test_df.apply(lambda x: [(x['time_list'][i]-x['time_list'][i-1])/24
                                                for i in range(1,x['tweet_amount_'+str(GATHER_POINT)+'.0h'])],
                                     axis=1)
test_df['len'] = test_df['delta_t'].map(len)
test_df = test_df[test_df['len']>0]
test_df = test_df.reset_index(drop=True)
test_news = test_df[['url','label']].merge(test_news[['url','vector']], on='url', how='left')

X_test = test_df.iloc[:, list(range(1, int(GATHER_POINT / INTERVAL)+2))].values
X_test2 = list(test_news['vector'])
y_test = list(test_df['label'])
dy_test = np.array(y_test).reshape((-1,1))-np.array(pred_event_test)


# In[ ]:
print(dy_train.shape) 

# time boosting
X_tr, X_val, y_tr, y_val = train_test_split(X_train, dy_train, test_size=0.1, random_state=42)

c_l = []
for i in range(dy_train.shape[1]):
    obj = 'reg:linear'
    c  = xgt_regressor(lr, max_d, estimators, X_tr, X_val, y_tr[:,i], y_val[:,i], obj)
    
    with open(path_log, "a") as text_file:
        text_file.write("\n\n ------ Parameters and Evaluation Results ------- \n")
        text_file.write("time boosting : \n")
        text_file.write("objective type : %s \n"%(obj))
        text_file.write("learning rate : %.4f \n"%(c[1][0]))
        text_file.write("max depth : %.4f \n"%(c[1][1]))
        text_file.write("number of estimators : %.4f \n\n"%(c[1][2]))
        
    c_l.append(c)

'''pred_time_train = []
for i in range(dy_train.shape[1]):
    pred_time_train.append(c_l[i][0].predict(X_train)+np.array(pred_event_train)[:,i])
pred_time_train = np.array(pred_time_train).transpose()
dy_train2 = np.array(y_train).reshape((-1,1))-np.array(pred_time_train)
'''
pred_time_test = []
for i in range(dy_test.shape[1]):
    pred_time_test.append(c_l[i][0].predict(X_test)+np.array(pred_event_test)[:,i])
pred_time_test = np.array(pred_time_test).transpose()

'''
dy_test2 = np.array(y_test).reshape((-1,1))-np.array(pred_time_test)


# In[56]:
print('time boosting finished!')

# news boosting
X_tr, X_val, y_tr, y_val = train_test_split(X_train2, dy_train2, test_size=0.1, random_state=42)

c_l = []
for i in range(dy_train2.shape[1]):
    obj = 'reg:linear'
    c  = xgt_regressor(lr, max_d, estimators, X_tr, X_val, y_tr[:,i], y_val[:,i], obj)
    
    with open(path_log, "a") as text_file:
        text_file.write("\n\n ------ Parameters and Evaluation Results ------- \n")
        text_file.write("news boosting : \n")
        text_file.write("objective type : %s \n"%(obj))
        text_file.write("learning rate : %.4f \n"%(c[1][0]))
        text_file.write("max depth : %.4f \n"%(c[1][1]))
        text_file.write("number of estimators : %.4f \n\n"%(c[1][2]))
        
    c_l.append(c)
    
pred_news_test = []
for i in range(dy_test2.shape[1]):
    pred_news_test.append(c_l[i][0].predict(X_test2)+np.array(pred_time_test)[:,i])
pred_news_test = np.array(pred_news_test).transpose()

'''
# In[ ]:


# calculate results
rmse, mae, mape = pred_eval(pred_time_test.sum(1)/30, y_test)
# np.save(path_result+name+"_output.npy", pred_news_test)

with open(path_log, "a") as text_file:
    text_file.write("\n\n ------ Evaluation Results ------- \n")

    text_file.write("best rmse : %.4f \n"%(rmse))
    text_file.write("best mae : %.4f \n"%(mae))
    text_file.write("best mape : %.4f \n"%(mape))

