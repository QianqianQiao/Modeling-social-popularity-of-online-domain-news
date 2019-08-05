
# coding: utf-8

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import pickle as pickle
import os
import random
import time
from sklearn.model_selection import train_test_split 
from fastprogress import master_bar, progress_bar
import argparse
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import seed_everything, pred_eval
from base_models import RMTPP
SEED = 2019

parser = argparse.ArgumentParser()
parser.add_argument("--gather_point", "-p", help="gather point, in hours", type=int, default = 5)
parser.add_argument("--mode", "-m", help="train, predict", type=str, default = 'predict')
parser.add_argument('--simu_time','-st', help='time of simulation', type=int, default=30)
parser.add_argument("--path_model", "-model", help="path", type=str, default = '../results/neuralpp/models/lstm:_bz:32_lr:0.001_hz:32_wd:0.pth')
args = parser.parse_args()

GATHER_POINT = args.gather_point
name = str(GATHER_POINT)
path_data = "../data/"
path_log  = "../results/neuralpp/"+name+"_log.txt"
path_result = "../results/neuralpp/"
path_models = "../results/neuralpp/models/"


# data organization
#train
if 1:
    train_df = pd.read_pickle(path_data + 'train.p')
    train_df['tweet_amount_5.0h'] = train_df['tw_vector'].map(len)
    train_df['event_feature'] = train_df.apply(lambda x: [list(x['tw_vector'][i]) + [x['time_list'][i]/24]
                                                          for i in range(0, x['tweet_amount_'+str(GATHER_POINT)+'.0h']-1)],
                                               axis = 1)
    train_df['delta_t'] = train_df.apply(lambda x: [(x['time_list'][i]-x['time_list'][i-1])/24
                                                    for i in range(1,x['tweet_amount_'+str(GATHER_POINT)+'.0h'])],
                                         axis=1)
    train_df['len'] = train_df['delta_t'].map(len)
    train_df = train_df[train_df['len']>0]
    train_df = train_df.reset_index(drop=True)
    print('len of training data:', len(train_df))
    train_es = list(train_df['event_feature'])
    train_delta = list(train_df['delta_t'])
    train_last = [i[j-1] for i,j in
                 zip(list(train_df['time_list']),
                     list(train_df['tweet_amount_'+str(GATHER_POINT)+'.0h']))]
    train_base = list(train_df['tweet_amount_'+str(GATHER_POINT)+'.0h'])
    train_label = list(train_df['label'])
#test
if args.mode == 'predict':
    test_df = pd.read_pickle(path_data + 'test.p')
    test_df['tweet_amount_5.0h'] = test_df['tw_vector'].map(len)
    test_df['event_feature'] = test_df.apply(lambda x: [list(x['tw_vector'][i]) + [x['time_list'][i]/24]
                                                          for i in range(0, x['tweet_amount_'+str(GATHER_POINT)+'.0h']-1)],
                                               axis = 1)
    test_df['delta_t'] = test_df.apply(lambda x: [(x['time_list'][i]-x['time_list'][i-1])/24
                                                    for i in range(1,x['tweet_amount_'+str(GATHER_POINT)+'.0h'])],
                                         axis=1)
    test_df['len'] = test_df['delta_t'].map(len)
    test_df = test_df[test_df['len']>0]
    test_df = test_df.reset_index(drop=True)
    print('number of testing data:', len(test_df))
    test_es = list(test_df['event_feature'])
    test_delta = list(test_df['delta_t'])
    test_last = [i[j-1] for i,j in
                 zip(list(test_df['time_list']),
                     list(test_df['tweet_amount_'+str(GATHER_POINT)+'.0h']))]
    test_base = list(test_df['tweet_amount_'+str(GATHER_POINT)+'.0h'])
    test_label = list(test_df['label'])


class PPTrainDataset(Dataset):
    def __init__(self, seq, delta_t):
        super().__init__()
        self.seq = seq
        self.delta_t = delta_t
        
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, idx):
        seq = self.seq[idx]
        delta_t = self.delta_t[idx]
                
        return seq, delta_t

def collate_fn(batch):
    data = [item[0] for item in batch]
    lens = [len(sq) for sq in data]
    padded_data = pad_sequence([torch.Tensor(v) for v in data], batch_first = True).cuda()
    target = [item[1] for item in batch]
    padded_target = pad_sequence([torch.Tensor(v) for v in target], batch_first = True).cuda()
    return padded_data, padded_target, lens

def lambda_t(past, w, t, t_j):
    dt = t-t_j
    if dt <0:
        dt = 0
    return np.exp(past + w * dt)

def simulate(past, w, N_simulation, last_time, base): # last_time load
    PREDICT_POINT = 24
    count = []
    count_single = []
    for j in range(N_simulation):
        t = args.gather_point
       # t = 5
        t_sample = []
        while t < PREDICT_POINT:
    #        print(t - last_time)
            lambda_star = lambda_t(past[-1], w[0], t, last_time)+0.0001
            u = np.random.rand()
            tao = -np.log(u) / lambda_star
            t += tao
            s = np.random.rand()
            if s <= lambda_t(past[-1], w[0], t, last_time) / lambda_star:
                t_sample.append(t)
        count_single.append(len(t_sample)+base)
    return count_single

def train_model(series_train, delta_train, batch_size, lr, hidden_size, weight_decay):
    batch_size = batch_size
    valid_batch_size = 32
    num_epochs = 200
    num_workers = 0

    # series settings
    feature_size = len(series_train[0][0])
    hidden_size = hidden_size
    dropout = 0
    lr = lr
    weight_decay = 0
    
    series_trn, series_val, delta_trn, delta_val = train_test_split(series_train, delta_train, test_size=0.1, random_state=SEED)
    train_dataset = PPTrainDataset(series_trn, delta_trn)
    valid_dataset = PPTrainDataset(series_val, delta_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn, num_workers = num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False,collate_fn=collate_fn, num_workers = num_workers)
                              
    model = RMTPP(feature_size, hidden_size, dropout)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
    
    best_epoch = -1
    best_loss = 0.
    l_list = []
    mb = master_bar(range(num_epochs))

    for epoch in mb:
        start_time = time.time()
        # train model
        model.train()
        avg_loss = 0.
        for series_batch, delta_batch, lens_batch in progress_bar(train_loader, parent=mb):
            loss, _, _ = model(series_batch, delta_batch, lens_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        #evaluate model
        model.eval()
        avg_val_loss = 0.
        for i, (series_batch, delta_batch, lens_batch) in enumerate(valid_loader):
            loss, _, _ = model(series_batch, delta_batch, lens_batch)
            avg_val_loss += loss.item() / len(valid_loader)
        l_list.append(avg_val_loss)
        
        if (epoch + 1) % 1 == 0:
            elapsed = time.time() - start_time
            mb.write(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        
        if avg_val_loss < best_loss:
            best_epoch = epoch + 1
            best_loss = avg_val_loss
            torch.save(model.state_dict(), path_models + name + '.pth')
            
    return best_epoch,best_loss


def predict_model(series_test, delta_test, batch_size, last_time, base, label, path_model):
    batch_size = batch_size
    num_workers = 0
    
    feature_size = len(series_test[0][0])
    hidden_size = 32
    dropout = 0
    
    test_dataset = PPTrainDataset(series_test, delta_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn, num_workers = num_workers)
    
    model = RMTPP(feature_size, hidden_size, dropout)
    model.load_state_dict(torch.load(path_model))
    model.cuda()
    model.eval()
    
    l_pred = []
    for i, (series_batch, delta_batch, lens_batch) in enumerate(test_loader):
        _, past, w_t = model(series_batch, delta_batch, lens_batch)
        p = past.squeeze(0).detach().cpu().numpy()
        w = w_t.detach().cpu().numpy()
        pred = simulate(p, w, args.simu_time, last_time[i], base[i])
        l_pred.append(pred)
        if i % 1000 == 0:
            print('done for ', i)
    return np.array(l_pred)


if args.mode == 'train':
    bs = [32]
    lr = [1e-3]
    hz = [32]
    wd = [0]

    for a in bs:
        for b in lr:
            for c in hz:
                for d in wd:
                    print ('------------------')
                    print ('model parameters:')
                    print ('batch size:',a)
                    print ('learning rate:',b)
                    print ('hidden size:', c)
                    print ('weight decay:', d)
                    print ('------------------')
                    be, bl = train_model(train_es, train_delta, a, b, c, d)
                    
                    with open(path_log, "a") as text_file:
                        text_file.write("\n\n ------ Trainging logs ------- \n")
                        text_file.write("batch size : %.4f \n"%(a))
                        text_file.write("learning rate : %.4f \n"%(b))
                        text_file.write("hidden size : %.4f \n"%(c))
                        text_file.write("best epoch :%.4f \n"%(be))
                        text_file.write("best loss : %.4f\n"%(bl))
    
elif args.mode == 'predict':
    result = predict_model(test_es, test_delta, 1,
            test_last, test_base, test_label, args.path_model)
    rmse, mae, mape = pred_eval(result.sum(1)/args.simu_time, test_label)
    np.save(path_result+name+"_output.npy", result)
    
    result_t = predict_model(train_es, train_delta, 1,
            train_last, train_base, train_label, args.path_model)
    np.save(path_result+name+"_train_output.npy", result_t)

    with open(path_log, "a") as text_file:
        text_file.write("\n\n ------ Evaluation Results ------- \n")

        text_file.write("best rmse : %.4f \n"%(rmse))
        text_file.write("best mae : %.4f \n"%(mae))
        text_file.write("best mape : %.4f \n"%(mape))













