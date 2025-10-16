#pylint: disable=E1101
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class multiTimeAttention(nn.Module):
    # https://github.com/reml-lab/mTAN
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn
    
    
    def forward(self, query, key, value, mask=None, dropout=None):
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)

        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)

        return self.linears[-1](x)
    
class ATE(nn.Module):
    def __init__(self, ori_input_dim, static_dim, nhidden=16, embed_time=16, num_heads=1, 
                 learn_emb=True, device='cuda', n_classes=2, static=True):
        super(ATE, self).__init__()
        self.embed_time = embed_time
        self.dim = ori_input_dim
        self.device = device
        self.nhidden = nhidden
        self.static_dim = static_dim
        self.static_emb = 8

        ## Define learning query (reference points)
        self.query = nn.Parameter(torch.linspace(0., 1., self.embed_time))

        self.learn_emb = learn_emb
        
        self.att_enc = multiTimeAttention(2*self.dim, self.dim, embed_time, num_heads)

        self.encoder = nn.GRU(self.dim, nhidden)

        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)

        if static:
            self.static_encoder = nn.Linear(self.static_dim, self.static_emb)
            d_fi = self.nhidden + self.static_emb
        else:
            d_fi = self.nhidden

        self.classifier = nn.Sequential(
            nn.Linear(d_fi, d_fi),
            nn.BatchNorm1d(d_fi),
            nn.GELU(),
            nn.Linear(d_fi, n_classes))

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    
    def fixed_time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        div_term = div_term
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def random_masking(self, irr_mask):
        mask_ratio = random.randint(1, 9) * 0.1
        temp = torch.full((irr_mask.size(0), irr_mask.size(1)), mask_ratio)
        temp = (1 - torch.bernoulli(temp)).cuda()
        temp = temp.repeat_interleave(irr_mask.size(-1), dim=-1).view(irr_mask.size())
        mask = irr_mask.clone() * temp
        return mask
    
    def forward(self, x, time_steps, static_info):
        irr_mask = x[:, :, self.dim:]
        
        mask = self.random_masking(irr_mask)
        
        irr_mask = torch.cat((irr_mask, irr_mask), 2)
        mask = torch.cat((mask, mask), 2)

        # Derive time embedding vectors for queries and keys
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
            query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key = self.fixed_time_embedding(time_steps.cpu(), self.embed_time).to(self.device)
            query = self.fixed_time_embedding(self.query.unsqueeze(0).cpu(), self.embed_time).to(self.device)
        
        # Generate repersentations for reference time points
        out = self.att_enc(query, key, x, irr_mask)
        out_te = out.clone()

        # Obtain outer product matrices for inputs and representations
        sx = x[:, :, :self.dim].clone()
        sx = torch.round(torch.sigmoid(torch.bmm(sx.transpose(-2, -1), sx)))
        sout = torch.sigmoid(torch.bmm(out_te.transpose(-2, -1), out_te))
        
        # Feed to GRU
        out = out.permute(1, 0, 2)
        _, out = self.encoder(out)
        out = out.permute(1, 0, 2)

        # Generate a masked context view for reference time points
        out1 = self.att_enc(query, key, x, mask)
        out1_te = out1.clone()

        if static_info is not None:
            static_out = self.static_encoder(static_info)
            cls_out = torch.cat([out.squeeze(), static_out], dim=1)
        else:
            cls_out = out.squeeze()

        return out_te, out1_te, self.classifier(cls_out), sx, sout, self.query