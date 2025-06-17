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
    
class Add_and_Norm(nn.Module):
    def __init__(self, dropout:float):
        super().__init()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self,x,sublayer):
        return self.norm(x + self.dropout(sublayer(x)))
    
class EncoderLayer(nn.Module):
    def __init__(self,self_attention_block: MultiheadAttention, feed_forward: FeedForward,dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward = feed_forward
        self.add_norm_layers = nn.ModuleList([Add_and_Norm(dropout) for _ in range(2)])

    def forward(self,x,src_mask):
        x = self.add_norm_layers[0](x,lambda x:self.self_attention_block(x,x,x,src_mask))
        x = self.add_norm_layers[1](x,self.feed_forward)
        return x

class EncoderStack(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.Encoder_layers = layers

    def forward(self,x,mask):
        for Encoder_layer in self.Encoder_layers:
            x = Encoder_layer(x,mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self,self_attention_block: MultiheadAttention,cross_attention_block: MultiheadAttention,feedforward:FeedForward,dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feedforward = feedforward
        self.add_norm_layers = nn.ModuleList([Add_and_Norm(dropout) for _ in range(3)])

    def forward(self,x,encoder_output,src_mask,target_mask):
        x = self.add_norm_layers[0](x,lambda x:self.self_attention_block(x,x,x,target_mask))
        x = self.add_norm_layers[1](x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.add_norm_layers[2](x,self.feedforward)
        return x

class DecoderStack(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.decoder_layers = layers
    
    def forward(self,x,encoder_output,src_mask,target_mask):
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x,encoder_output,src_mask,target_mask)
        return x

class ProjectionLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.projection = nn.Linear(d_model,vocab_size)
    
    def forward(self,x):
        return torch.log_softmax(self.projection(x),dim=-1)

class Transformer(nn.Module):
    def __init__(self,encoder:EncoderStack, decoder:DecoderStack,input_embedding:InputEmbeddings,inputPos_embedding:PositionalEmbeddings,output_embedding:InputEmbeddings,outputPos_embedding:PositionalEmbeddings, projection_layer:ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_embedding = input_embedding
        self.input_pos = inputPos_embedding
        self.output_embedding = output_embedding
        self.output_pos = outputPos_embedding
        self.projection_layer = projection_layer
    
    def encode(self,input,src_mask):
        input = self.input_embedding(input)
        input = self.input_pos(input)
        return self.encoder(input,src_mask)
    
    def decode(self,output,encoder_output,src_mask,target_mask):
        output = self.output_embedding(output)
        output = self.output_pos(output)
        return self.decoder(output,encoder_output,src_mask,target_mask)
    
    def project(self,x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size:int, target_vocab_size:int, src_seq_len:int, target_seq_len:int, d_model:int=512, N:int=6, head_num:int = 6, dropout:float = 0.1,hidden_dim:int = 2048):
    #Creating embedding layers
    input_embed = InputEmbeddings(d_model,src_vocab_size)
    output_embed = InputEmbeddings(d_model,target_vocab_size)

    #Creating Position embedding layers
    input_pos = PositionalEmbeddings(d_model,src_seq_len,dropout)
    output_pos = PositionalEmbeddings(d_model,target_seq_len,dropout)

    #Create Encoder Stack list containing N encoding layers
    encoder_stack = []
    for _ in range(N):
        enc_self_att = MultiheadAttention(d_model,head_num,dropout)
        enc_feed_forward = FeedForward(d_model,hidden_dim,dropout)
        encoder_layer = EncoderLayer(enc_self_att,enc_feed_forward,dropout)
        encoder_stack.append(encoder_layer)

    #Creating Decoder Stack list having N decoding layers
    decoder_stack = []
    for _ in range(N):
        dec_self_att = MultiheadAttention(d_model,head_num,dropout)
        dec_cross_att = MultiheadAttention(d_model,head_num,dropout)
        dec_feed_forward = FeedForward(d_model,hidden_dim,dropout)
        decoder_layer = DecoderLayer(dec_self_att,dec_cross_att,dec_feed_forward,dropout)
        decoder_stack.append(decoder_layer)

    #Creating final encoder and decoder...normal list to nn.Modulelist
    encoder = EncoderStack(nn.ModuleList(encoder_stack))
    decoder = DecoderStack(nn.ModuleList(decoder_stack))

    #Creating Projection layer
    projection_layer = ProjectionLayer(d_model,target_vocab_size)

    #Creating Transformer
    transformer = Transformer(encoder,decoder,input_embed,input_pos,output_embed,output_pos,projection_layer)

    #initialize parameters
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    
    return transformer