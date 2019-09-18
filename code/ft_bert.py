
# coding: utf-8

# In[1]:

import os
import random
import numpy as np
import pandas as pd
import random
import pickle as pickle
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# In[2]:

'''
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


# In[5]:


train_df['tokens'] = train_df['sentences'].apply(lambda x: [tokenizer.tokenize('[CLS] '+ i +' [SEP]') for i in x])
train_df['indexed_tokens'] = train_df['tokens'].apply(lambda x: [tokenizer.convert_tokens_to_ids(i) for i in x])


# In[11]:


test_df['tokens'] = test_df['sentences'].apply(lambda x: [tokenizer.tokenize('[CLS] '+ i +' [SEP]') for i in x])
test_df['indexed_tokens'] = test_df['tokens'].apply(lambda x: [tokenizer.convert_tokens_to_ids(i) for i in x])


# In[12]:


train_df.to_pickle('./data/train_token.p')
test_df.to_pickle('./data/test_token.p')


# # Fine-tune model

# In[ ]:

'''
train_df = pd.read_pickle('./data/train_token.p')
test_df = pd.read_pickle('./data/test_token.p')


# In[13]:

#train_df = train_df.iloc[:1,:]


# In[ ]:

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

def change_dropout(module,p):
    for name in module.__dict__['_modules']:
        if name == "dropout":
            module.__dict__['_modules'][name].p = p
        else:
            change_dropout(module.__dict__['_modules'][name],p)


# In[21]:


def train_model(train_data, test_data, batch_size, lr, dropout, epoch):
    seed_everything(2019)
    print('number of training data: ', len(train_data))
    # train:
    sub_batch = batch_size
    EPOCHS = epoch

    loss_c = nn.MSELoss().cuda()
    bert = BertModel.from_pretrained('bert-base-cased').cuda()
    change_dropout(bert, dropout) 
    ln = nn.Linear(768, 1).cuda()
    opt = torch.optim.Adam(list(bert.parameters())+list(ln.parameters()), lr=lr)
    
    train_indexed_tokens = list(train_data['indexed_tokens']) 
    train_label = list(train_data['label'])
    
    for e in range(1, EPOCHS+1):
        for b in range(0,len(train_data),sub_batch):
            start = b
            end = min(len(train_data),b+sub_batch)
            
            label = torch.tensor(train_label[start:end]).float().cuda()
            text_vec = []
            for indexed_tokens in train_indexed_tokens[start:end]: # for each news:
                segment_ids = [[0]*len(i) for i in indexed_tokens]
                tokens_tensor = [torch.tensor([i]).cuda() for i in indexed_tokens]
                segments_tensors = [torch.tensor([i]).cuda() for i in segment_ids]
                
                sent_vec = []
                for i, j in zip(tokens_tensor, segments_tensors): # for each sents in this news:
                    encoded_layers, _ = bert(i, j, output_all_encoded_layers=False)
                    sent_vec.append(encoded_layers.squeeze(0)[0,:])
                text_vec.append(torch.stack(sent_vec, dim=0).sum(0))
            pred = F.elu(ln(torch.stack(text_vec, dim=0))).squeeze(-1)
            
            loss = loss_c(pred, label)
            loss.backward()
            opt.step()
            opt.zero_grad()
          #  print(b) 
            if b % 1000 == 0:
                print ("Epoch =",e,"loss = ",loss.item())
    torch.save(model.state_dict(), 'ft_bert.pth')
    print('Training complete!')    
    torch.cuda.empty_cache()

    # test:
    bert.eval()
    ln.eval()
    test_indexed_tokens = list(test_data['indexed_tokens']) 
    test_label = list(test_data['label'])
    pred = []
    with torch.no_grad():
        text_vec = []
        for indexed_tokens in test_indexed_tokens: # for each news:
            segment_ids = [[0]*len(i) for i in indexed_tokens]
            tokens_tensor = [torch.tensor([i]).cuda() for i in indexed_tokens]
            segments_tensors = [torch.tensor([i]).cuda() for i in segment_ids]

            sent_vec = []
            for i, j in zip(tokens_tensor, segments_tensors): # for each sents in this news:
                encoded_layers, _ = bert(i, j, output_all_encoded_layers=False)
                sent_vec.append(encoded_layers.squeeze(0)[0,:])
            text_vec = torch.stack(sent_vec, dim=0).sum(0)
            pred.append(ln(text_vec).squeeze(-1).item())
        a, b, c = pred_eval(pred, test_label)

        print('rmse is:', a)
        print('mae is:', b)
        print('mape is:', c)




# In[ ]:


train_model(train_df, test_df, batch_size=6, lr=5e-5, dropout=0.15, epoch=1)

