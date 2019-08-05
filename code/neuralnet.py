
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
from base_models import CoModel
SEED = 2019

parser = argparse.ArgumentParser()

# global parameters
parser.add_argument("--mode", "-m", help="train, predict,search", type=str, default = 'predict')
parser.add_argument("--gather_point", "-p", help="gather point, in hours", type=int, default = 5)
parser.add_argument("--model_type", "-mt", help="model type", type=str, default = 'concatenate')

# training parameters
parser.add_argument("--hidden_size", "-hz", help="hidden size", type=int, default = 32)
parser.add_argument("--batch_size", "-bz", help="batch size", type=int, default = 16)
parser.add_argument("--lr", "-lr", help="learning rate", type=float, default = 1e-3)
parser.add_argument("--weight_decay", "-wd", help="weight decay", type=float, default = 0)

# predicting parameters
# hidden size
parser.add_argument("--predict_models", "-models", help="txt with list of predict models", type=str, default = '0.txt')

args = parser.parse_args()
GATHER_POINT = args.gather_point
INTERVAL = 0.25 # in hours
name = str(GATHER_POINT) + args.model_type
path_data = "../data/"
path_log  = "../results/neuralnet/"+name+"_log.txt"
path_result = "../results/neuralnet/"
path_models = "../results/neuralnet/models/" + name + '/'

if not os.path.exists(path_models):
    os.makedirs(path_models)

if args.mode == 'predict':
    path_list = []
    with open(path_models+args.predict_models) as f:
        for line in f:
            path_list.append(line.strip())

# data organization
#train

if args.mode == 'train' or 'search':
    train_df = pd.read_pickle(path_data + 'train.p')
    train_news = pd.read_pickle(path_data+'train_news.p')
    train_df.drop(index = 12797, inplace=True)
    train_news.drop(index = 12797, inplace = True)
    train_df['tweet_amount_5.0h'] = train_df['tw_vector'].map(len)
    train_ts = train_df.iloc[:, list(range(1, int(GATHER_POINT / INTERVAL)+2))].values
    train_es = [i[:j] for i,j in
                 zip(list(train_df['tw_vector']),
                     list(train_df['tweet_amount_'+str(GATHER_POINT)+'.0h']))]
    train_es_time = [i[:j] for i,j in
                 zip(list(train_df['time_list']),
                     list(train_df['tweet_amount_'+str(GATHER_POINT)+'.0h']))]
    train_ns = list(train_news['vector'])
    train_label = list(train_df['label'])
    train_data = tuple(zip(train_ts, train_es, train_es_time, train_ns))

#test
test_df = pd.read_pickle(path_data + 'test.p')
test_news = pd.read_pickle(path_data+'test_news.p')
test_df['tweet_amount_5.0h'] = test_df['tw_vector'].map(len)
test_ts = test_df.iloc[:, list(range(1, int(GATHER_POINT / INTERVAL)+2))].values
test_es = [i[:j] for i,j in
             zip(list(test_df['tw_vector']),
                 list(test_df['tweet_amount_'+str(GATHER_POINT)+'.0h']))]
test_es_time = [i[:j] for i,j in
             zip(list(test_df['time_list']),
                 list(test_df['tweet_amount_'+str(GATHER_POINT)+'.0h']))]
test_ns = list(test_news['vector'])
test_label = list(test_df['label'])
test_data = tuple(zip(test_ts, test_es,test_es_time, test_ns))

class CoModelDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq_ts = self.data[idx][0]
        seq_es = torch.tensor(self.data[idx][1]).cuda()
        seq_es_t = torch.tensor(self.data[idx][2]).unsqueeze(-1).cuda()
        news = self.data[idx][3]
        label = self.label[idx]
        return seq_ts, seq_es, seq_es_t, news, label

def collate_fn(batch):
    ts = torch.tensor([item[0] for item in batch]).float().unsqueeze(-1).cuda()
    es = [item[1] for item in batch]
    lens = [len(sq) for sq in es]
    padded_es = pad_sequence(es, batch_first = True).float().cuda()
    es_t = [item[2] for item in batch]
    padded_es_t = pad_sequence(es_t, batch_first = True).float().cuda()
    news = torch.tensor([item[3] for item in batch]).float().cuda()
    label = torch.tensor([item[4] for item in batch]).float().cuda()
    return ts, padded_es, padded_es_t, lens, news, label

seed_everything(SEED)
def train_model(train_data, train_label, test_data, test_label, model_type, batch_size,
                hidden_size, weight_decay, lr=1e-3):
    valid_batch_size = 32
    num_epochs = 200
    num_workers = 0

    # series settings
    features_ts = 1
    features_es = train_data[0][1][0].shape[0]
    features_ns = train_data[0][3].shape[0]

    hidden_size_time = 4
    
    data_trn, data_val, label_trn, label_val = train_test_split(train_data, train_label, test_size=0.1, random_state=SEED)

    train_dataset = CoModelDataset(data_trn, label_trn)
    valid_dataset = CoModelDataset(data_val, label_val)
    test_dataset = CoModelDataset(test_data, test_label)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn, num_workers = num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False,collate_fn=collate_fn, num_workers = num_workers)
    test_loader = DataLoader(test_dataset, batch_size=valid_batch_size, shuffle=False,collate_fn=collate_fn, num_workers = num_workers)

    model = CoModel(features_ts, features_es, features_ns, hidden_size_time, hidden_size, model_type)
    model.cuda()
    
    para_reg = []
    para_others = []
    for name, p in model.named_parameters():
        if 'fc_g' in name:
            para_reg.append(p)
        else:
            para_others.append(p)
            
    optimizer = optim.Adam([{'params': para_reg, 'weight_decay':weight_decay},
              {'params': para_others}
              ], lr=lr)
    
    criterian = nn.MSELoss().cuda()
    
    best_epoch = -1
    best_loss = 100000
    l_list = []
    mb = master_bar(range(num_epochs))

    for epoch in mb:
        start_time = time.time()
        
        # train model
        model.train()
        avg_loss = 0.
        for ts_batch, es_batch, es_t_batch, lens_batch, news_batch, label_batch in progress_bar(train_loader, parent=mb):
            pred = model(ts_batch, es_batch, es_t_batch, lens_batch, news_batch, model_type)
            loss = criterian(pred, label_batch)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
            
        #evaluate model
        model.eval()
        avg_val_loss = 0.
        for i, (ts_batch, es_batch, es_t_batch, lens_batch, news_batch, label_batch) in enumerate(valid_loader):
            pred = model(ts_batch, es_batch, es_t_batch, lens_batch, news_batch, model_type)
            loss = criterian(pred, label_batch)
        
            avg_val_loss += loss.item() / len(valid_loader)
        l_list.append(avg_val_loss)
        
        if (epoch + 1) % 1 == 0:
            elapsed = time.time() - start_time
            with open(path_log, "a") as text_file:
                text_file.write(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
                text_file.write('\n')
            mb.write(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')

            
        if avg_val_loss < best_loss:
            best_epoch = epoch + 1
            best_loss = avg_val_loss
        if args.mode == 'train':
            if epoch>150:
                torch.save(model.state_dict(),
                           path_models
                           +'_epoch:'+str(epoch+1)
                           +'_trainl:'+str(round(avg_loss,2))
                           +'_validl:'+str(round(avg_val_loss,2))+'.pth')
    return best_epoch, best_loss


def predict_model(test_data, test_label, model_type, hidden_size, path_list):
    batch_size = 1
    num_workers = 0
    
    # series settings
    features_ts = 1
    features_es = test_data[0][1][0].shape[0]
    features_ns = test_data[0][3].shape[0]
    hidden_size_time = 4
    
    test_dataset = CoModelDataset(test_data, test_label)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn, num_workers = num_workers)

    model = CoModel(features_ts, features_es, features_ns, hidden_size_time, hidden_size, model_type)

    ll_pred = []
    for path in path_list:
        model.load_state_dict(torch.load(path_models+path))
        model.cuda()
        model.eval()
        l_pred = []
        for i, (ts_batch, es_batch, es_t_batch, lens_batch, news_batch, label_batch) in enumerate(test_loader):
            pred = model(ts_batch, es_batch, es_t_batch, lens_batch, news_batch, model_type)
            l_pred.append(pred.item())
        ll_pred.append(np.array(l_pred))
    test_pred = np.array(ll_pred).sum(0)/len(ll_pred)

    return test_pred

if args.mode == 'search':
    bs = [16]
    lr = [7e-3,5e-3,3e-3,1e-3]
    hz = [8,4]
    wd = [0]

    for a in bs:
        for b in lr:
            for c in hz:
                for d in wd:
                    print ('------------------')
                    print('time size == 4')
                    print ('model parameters:')
                    print ('batch size:',a)
                    print ('learning rate:',b)
                    print ('hidden size:', c)
                    print ('weight decay:',d)
                    print ('------------------')
                    be, bl = train_model(train_data, train_label, test_data, test_label, args.model_type,a,c,d,b)
                        
                    with open(path_log, "a") as text_file:
                        text_file.write("\n\n ------ Searching logs ------- \n")
                        text_file.write("batch size : %.4f \n"%(a))
                        text_file.write("learning rate : %.4f \n"%(b))
                        text_file.write("hidden size : %.4f \n"%(c))
                        text_file.write("weight decay : %.4f \n"%(d))
                        text_file.write("best epoch : %d \n"%(be))
                        text_file.write("best loss : %.4f \n\n"%(bl))
                        text_file.write('----------------------------------------------------------------\n')

elif args.mode == 'train':
    be, bl = train_model(train_data, train_label, test_data, test_label, args.model_type,
                         args.batch_size,args.hidden_size, args.weight_decay, args.lr)
    with open(path_log, "a") as text_file:
        text_file.write("\n\n ------ Training logs ------- \n")
        text_file.write("batch size : %.4f \n"%(args.batch_size))
        text_file.write("learning rate : %.4f \n"%(args.lr))
        text_file.write("hidden size : %.4f \n"%(args.hidden_size))
        text_file.write("weight decay : %.4f \n"%(args.weight_decay))
        text_file.write("best epoch : %d \n"%(be))
        text_file.write("best loss : %.4f \n\n"%(bl))
        text_file.write('----------------------------------------------------------------------\n')

elif args.mode == 'predict':
    result = predict_model(test_data, test_label, args.model_type,
            args.hidden_size, path_list)
    rmse, mae, mape = pred_eval(result, test_label)
    np.save(path_result+name+"_output.npy", result)
    print('rmse is:', rmse)
    print('mae is:', mae)
    print('mape is:', mape)

    with open(path_log, "a") as text_file:
        text_file.write("\n\n ------ Evaluation Results ------- \n")

        text_file.write("best rmse : %.4f \n"%(rmse))
        text_file.write("best mae : %.4f \n"%(mae))
        text_file.write("best mape : %.4f\n \n"%(mape))
        
        for i in path_list:
            text_file.write(i+'\n')
        text_file.write('----------------------------------------------------------------------\n')





