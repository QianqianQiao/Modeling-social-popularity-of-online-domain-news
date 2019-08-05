import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch.nn as nn
import pickle

with open('./data/tw_text.pkl', 'rb') as f:
    data = pickle.load(f)
# data = [['I am a girl','you are a boy'],['we are human'],['what is it']]

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# load model
model = BertModel.from_pretrained('bert-base-cased')
model.eval()
# model.cuda()

text_vector = []
for ind,text in enumerate(data):
    # convert sentence to tokens and create segment_vectors
    tokenized_text = [tokenizer.tokenize('[CLS] ' + sent + ' [SEP]') for sent in text]
    indexed_tokens = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
    segment_ids = [[0]*len(i) for i in indexed_tokens]
    tokens_tensor = [torch.tensor([i]).cuda() for i in indexed_tokens]
    segments_tensors = [torch.tensor([i]).cuda() for i in segment_ids]

    sent_vector = []
    # use last hidden state and average word vector as sentence vector
    with torch.no_grad():
        for i,j in zip(tokens_tensor, segments_tensors):
            encoded_layers, _ = model(i,j,output_all_encoded_layers=False)
            print(encoded_layers.size())
            sent_vector.append(encoded_layers.squeeze(0).sum(0).cpu().numpy())
    text_vector.append(sent_vector)
    print(len(sent_vector))

    # if ind % 100 == 0:
    #     print('%d news already been converted'%ind)

# np.save('./data/vector.npy', text_vector)
print(len(text_vector))
