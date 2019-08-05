
# coding: utf-8

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from utils import pred_eval
import numpy as np


class RMTPP(nn.Module):
 
    def __init__(self, feature_size, hidden_size, dropout, n_layers=1, n_directions=1):
        super(RMTPP, self).__init__()
        self.feature_size = feature_size
        self.n_directions = n_directions
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_size=feature_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers, dropout = dropout,  batch_first=True) #bidirectional=True
        self.bn = nn.BatchNorm1d(hidden_size)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        self.w_t = nn.Parameter(torch.rand(1, requires_grad=True).cuda())
        self.fc_vt = nn.Linear(hidden_size,1)
        
    def f_star(self, delta_t, hidden, mask):
        past = F.elu(self.fc_vt(hidden)).squeeze(-1) # batch x seq_len
        current = self.w_t * delta_t
        return (past + current + torch.exp(past)/(self.w_t) - 
                torch.exp(past+current)/(self.w_t)).mul(mask).sum(0).sum(0), past
    
    def forward(self, input_seq, delta_seq, lens):
#         lens = [len(i) for i in input_seq]
        max_len = max(lens)
        mask = torch.arange(max_len).expand(len(lens), max_len) < torch.tensor(lens).unsqueeze(1)
        mask = mask.float().cuda()
        pack_es = pack_padded_sequence(input_seq, lens, batch_first=True, enforce_sorted=False)

        output, _ = self.rnn(pack_es) # output: batch x seq_len x dim
        unpacked, _ = pad_packed_sequence(output, batch_first = True)
        fstar, past = self.f_star(delta_seq, unpacked, mask)
#         print(self.w_t)
        return -fstar, past, self.w_t

def xgt_regressor(lr, max_d, estimators, X_train, X_test, y_train, y_test, obj):
    rmse = 10000
    for i in lr:
        for j in max_d:
            for k in estimators:

                clf=XGBRegressor(learning_rate=i,
                                n_estimators=k,
                                max_depth=j, 
                                min_child_weight=1, 
                                gamma=1,
                                subsample=0.5,
                                colsample_bytree=0.8,
                                objective=obj,
                                nthread=4,
                                scale_pos_weight=1, 
                                missing=np.nan)
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                
                a,b,c = pred_eval(y_pred,y_test)

                if a < rmse:
                    b_lr = i
                    b_d = j
                    b_e = k
                    
                    rmse = a
                    clf_b = clf
        
    return clf_b, (b_lr, b_d, b_e)


class CoModel(torch.nn.Module):
    def __init__(self, features_ts, features_es, features_ns, hidden_size_time, hidden_size, model_type, n_layers=2):
        super(CoModel, self).__init__()

        self.rnn_ts = nn.LSTM(input_size=features_ts,
                            hidden_size=hidden_size,
                            num_layers=n_layers, batch_first=True)

        self.rnn_es = nn.LSTM(input_size=features_es+hidden_size_time,
                            hidden_size=hidden_size,
                            num_layers=n_layers, batch_first=True)
        
        self.fc_time = nn.Linear(1, hidden_size_time)
        self.fc_news = nn.Linear(features_ns, hidden_size)

        if model_type == 'concatenate':
            self.fc0 = nn.Linear(hidden_size*3, 1, bias=True)
        if model_type == 'mix1':
            self.fc1 = nn.Linear(hidden_size, 1)
            self.fc2 = nn.Linear(hidden_size, 1)
            self.fc3 = nn.Linear(hidden_size, 1)
            self.fc4 = nn.Linear(3, 1)
        if model_type == 'mix2' or 'mix3':
            self.fc = nn.Linear(hidden_size, 1)
            self.fc_g1 = nn.Linear(hidden_size, 1)
            self.fc_g2 = nn.Linear(hidden_size, 1)
            self.fc_g3 = nn.Linear(hidden_size, 1)
            
        for name, param in self.rnn_ts.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
                
        for name, param in self.rnn_es.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        
    def forward(self, input_ts, input_es, input_es_time, lens, input_news, model_type):
        seq_es = torch.cat([F.elu(self.fc_time(input_es_time)), input_es], -1)
        max_len = max(lens)
        mask = torch.arange(max_len).expand(len(lens), max_len) < torch.tensor(lens).unsqueeze(1)
        mask = mask.float().cuda()
        mask_ = mask.unsqueeze(-1).expand_as(seq_es)
        seq_es_ = seq_es.mul(mask_)
        
        input_es_pack = pack_padded_sequence(seq_es_, lens, batch_first=True,  enforce_sorted=False)
        _, (hn_ts,_) = self.rnn_ts(input_ts) 
        _, (hn_es,_) = self.rnn_es(input_es_pack)
        hn_ts = hn_ts.sum(0)
        hn_es = hn_es.sum(0)
        hn_ns = F.elu(self.fc_news(input_news))
        # print(self.fc.bias.size())
        if model_type == 'concatenate':
            hn = torch.cat([hn_ts, hn_es, hn_ns], -1) # [num_layers*num_directions x batch_szie x dim]
            #print(hn.size())
            return self.fc0(hn).squeeze(-1)
        if model_type == 'mix1':# non_softmax
            hn = torch.cat([F.elu(self.fc1(hn_ts)), F.elu(self.fc2(hn_es)),
                             F.elu(self.fc3(hn_ns))], -1) # [num_layers*num_directions x batch_size x dim]
            return self.fc4(hn).squeeze(-1)
        if model_type == 'mix2':# mix_pred: attn_share, pred_non_share(unstable)
            hn = torch.stack([hn_ts,hn_es,hn_ns], 0)
            attn = F.softmax(F.elu(self.fc(hn)),dim=0)
            pred = torch.stack([self.fc_g1(hn_ts), self.fc_g2(hn_es), self.fc_g3(hn_ns)], 0)
            return attn.mul(pred).sum(0).squeeze(-1)
        if model_type == 'mix3':# share_pred: both share
            hn = torch.stack([hn_ts,hn_es,hn_ns], 0)
            attn = F.softmax(F.elu(self.fc(hn)),dim=0)
            pred = F.elu(self.fc_g1(hn))
            return attn.mul(pred).sum(0).squeeze(-1)
