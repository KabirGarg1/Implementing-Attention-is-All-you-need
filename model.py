import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super.__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)  #Embedding layer weights multiplied by root of d_model

class PositionalEmbeddings(nn.Module):
    def __init__(self,d_model:int,seq_len:int,dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #making a matrix of Seq len times d_model cuz need (seq len) number of position vectors..each of size d_model
        pe = torch.zeros(seq_len,d_model)
        position = torch.arange(0,seq_len,dtype = torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0,seq_len,2).float() * math.log(10000.0) / d_model)

        pe[:,::2] = torch.sin(position/div)
        pe[:,1::2] = torch.cos(position/div)
        pe = pe.unsqueeze(1)

        self.register_buffer("pe",pe)
    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean = x.mean(dim = -1,keepdim = True)
        std = x.std(dim = -1,keepdim = True)   
        return self.alpha*(x-mean)/(std+self.eps)  + self.bias

class FeedForward(nn.Module):
    def __init__(self,d_model:int,hidden_dim:int,dropout:float):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(d_model,hidden_dim),
                                    torch.relu(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden_dim,d_model))
    def forward(self,x):
        return self.network(x)

class MultiheadAttention(nn.Module):
    def __init__(self, d_model:int, head_num:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.head_num = head_num
        self.dropout = nn.Dropout(dropout)

        assert d_model%head_num == 0,"d_model not divbisible by head_num"

        self.d_k = d_model//head_num
        self.W_q = nn.linear(d_model,d_model)
        self.W_k = nn.linear(d_model,d_model)
        self.W_v = nn.linear(d_model,d_model)
        self.W_o = nn.linear(d_model,d_model)

    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0,-1e9)
        attention_scores = attention_scores.softmax(-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value),attention_scores

    def forward(self,q,k,v,mask):
        query = self.W_q(q)
        key = self.W_k(k)
        value = self.W_v(v)

        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x,self.attention_scores = MultiheadAttention.attention(query,key,value,mask,self.dropout)
        x = x.transpose(1,2).contiguous.view(x.shape[0],-1,self.head_num*self.d_k)

        return self.W_o(x)
    
