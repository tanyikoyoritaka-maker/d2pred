"""Architecture-level benchmark reproductions. These are not official implementations."""

import math
import torch
import torch.nn as nn


class Regressor(nn.Module):
    def predict(self, features): return torch.sigmoid(self.head(features)).squeeze(-1)


class CNN(Regressor):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(1,32,5,padding=2),nn.ReLU(),nn.Conv1d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1)); self.head=nn.Linear(64,1)
    def forward(self,x): return self.predict(self.net(x.unsqueeze(1)).squeeze(-1))


class RNNModel(Regressor):
    def __init__(self,bidirectional=False):
        super().__init__(); self.rnn=nn.LSTM(1,64,2,batch_first=True,bidirectional=bidirectional,dropout=.1); self.head=nn.Linear(64*(2 if bidirectional else 1),1)
    def forward(self,x): y,_=self.rnn(x.unsqueeze(-1)); return self.predict(y[:,-1])


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dimension=64, maximum_length=2048):
        super().__init__()
        position=torch.arange(maximum_length,dtype=torch.float32).unsqueeze(1)
        divisor=torch.exp(torch.arange(0,dimension,2,dtype=torch.float32)*(-math.log(10000.0)/dimension))
        values=torch.zeros(maximum_length,dimension)
        values[:,0::2]=torch.sin(position*divisor); values[:,1::2]=torch.cos(position*divisor)
        self.register_buffer("values",values.unsqueeze(0),persistent=False)
    def forward(self,x):
        if x.size(1)>self.values.size(1): raise ValueError("Sequence exceeds positional encoding capacity")
        return x+self.values[:,:x.size(1)].to(dtype=x.dtype,device=x.device)


class TransformerModel(Regressor):
    """Order-aware encoder-only Transformer baseline for scalar RUL regression."""
    architecture_version="transformer_rul_order_aware_v2"
    def __init__(self):
        super().__init__(); self.embed=nn.Linear(1,64); self.position=SinusoidalPositionEncoding(64); layer=nn.TransformerEncoderLayer(64,4,128,.1,batch_first=True,norm_first=True); self.encoder=nn.TransformerEncoder(layer,2,norm=nn.LayerNorm(64)); self.pool=nn.Linear(64,1); self.head=nn.Linear(64,1)
    def forward(self,x):
        z=self.encoder(self.position(self.embed(x.unsqueeze(-1))))
        weights=torch.softmax(self.pool(z).squeeze(-1),dim=1)
        return self.predict((z*weights.unsqueeze(-1)).sum(1))
    def architecture_metadata(self):
        return {
            "implementation": type(self).__name__,
            "architecture_version": self.architecture_version,
            "source": "models.benchmark",
            "input": "univariate DI prefix resampled to 128 steps",
            "order_encoding": "fixed sinusoidal positional encoding",
            "encoder_layers": 2,
            "embedding_dimension": 64,
            "attention_heads": 4,
            "pooling": "learned temporal attention pooling",
            "output": "sigmoid-normalized scalar RUL",
        }


class TemporalBlock(nn.Module):
    def __init__(self,channels,dilation=1):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(channels,channels,3,padding=dilation,dilation=dilation),nn.ReLU(),nn.BatchNorm1d(channels),nn.Conv1d(channels,channels,3,padding=dilation,dilation=dilation),nn.ReLU())
    def forward(self,x): return x+self.net(x)


class TCNBiGRU(Regressor):
    def __init__(self):
        super().__init__(); self.inp=nn.Conv1d(1,32,3,padding=1); self.tcn=nn.Sequential(TemporalBlock(32,1),TemporalBlock(32,2),TemporalBlock(32,4)); self.gru=nn.GRU(32,32,batch_first=True,bidirectional=True); self.head=nn.Linear(64,1)
    def forward(self,x): y=self.tcn(self.inp(x.unsqueeze(1))).transpose(1,2); y,_=self.gru(y); return self.predict(y[:,-1])


class ATCN(Regressor):
    def __init__(self):
        super().__init__(); self.inp=nn.Conv1d(1,64,3,padding=1); self.tcn=nn.Sequential(TemporalBlock(64,1),TemporalBlock(64,2),TemporalBlock(64,4)); self.attn=nn.Linear(64,1); self.head=nn.Linear(64,1)
    def forward(self,x): y=self.tcn(self.inp(x.unsqueeze(1))).transpose(1,2); w=torch.softmax(self.attn(y),1); return self.predict((y*w).sum(1))


class MDFA(Regressor):
    def __init__(self):
        super().__init__(); self.stem=nn.Conv1d(1,32,3,padding=1); self.branches=nn.ModuleList([nn.Conv1d(32,32,3,padding=d,dilation=d) for d in (1,2,4,8)]); self.channel=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Conv1d(128,32,1),nn.ReLU(),nn.Conv1d(32,128,1),nn.Sigmoid()); self.time=nn.Conv1d(128,1,1); self.head=nn.Linear(128,1)
    def forward(self,x):
        z=torch.relu(self.stem(x.unsqueeze(1))); z=torch.cat([torch.relu(branch(z)) for branch in self.branches],1); z=z*self.channel(z); w=torch.softmax(self.time(z),-1); return self.predict((z*w).sum(-1))


class SparseAttentionBlock(nn.Module):
    """Pre-norm sparse attention block with an explicit fixed attention pattern."""
    def __init__(self, dimension=64, heads=4, local_radius=8, landmark_stride=None, dropout=.1):
        super().__init__(); self.local_radius=local_radius; self.landmark_stride=landmark_stride
        self.norm1=nn.LayerNorm(dimension); self.attention=nn.MultiheadAttention(dimension,heads,dropout=dropout,batch_first=True)
        self.norm2=nn.LayerNorm(dimension); self.ffn=nn.Sequential(nn.Linear(dimension,dimension*2),nn.GELU(),nn.Dropout(dropout),nn.Linear(dimension*2,dimension)); self.dropout=nn.Dropout(dropout)
    def attention_mask(self,length,device):
        index=torch.arange(length,device=device); distance=(index[:,None]-index[None,:]).abs()
        allowed=distance<=self.local_radius
        if self.landmark_stride:
            landmarks=(index % self.landmark_stride)==0
            allowed=allowed | landmarks[None,:] | torch.eye(length,dtype=torch.bool,device=device)
        return ~allowed
    def forward(self,x):
        z=self.norm1(x); attended,_=self.attention(z,z,z,attn_mask=self.attention_mask(x.size(1),x.device),need_weights=False)
        x=x+self.dropout(attended); return x+self.dropout(self.ffn(self.norm2(x)))


class SBiTransformer(Regressor):
    """SBi-Transformer architecture-level reproduction for univariate AC DI.

    It retains the paper-visible requirements: order encoding, two sparse
    attention layers for local/global context, decoder removal, BiLSTM local
    temporal modeling, and temporal attention regression.
    """
    architecture_version="sbi_transformer_sparse_bilstm_v2"
    def __init__(self):
        super().__init__(); self.embed=nn.Linear(1,64); self.position=SinusoidalPositionEncoding(64)
        self.local_sparse_attention=SparseAttentionBlock(64,4,local_radius=8)
        self.global_sparse_attention=SparseAttentionBlock(64,4,local_radius=4,landmark_stride=16)
        self.bilstm=nn.LSTM(64,48,batch_first=True,bidirectional=True)
        self.time_attn=nn.Sequential(nn.Linear(96,48),nn.Tanh(),nn.Linear(48,1)); self.head=nn.Linear(96,1)
    def forward(self,x):
        z=self.position(self.embed(x.unsqueeze(-1)))
        z=self.local_sparse_attention(z); z=self.global_sparse_attention(z)
        z,_=self.bilstm(z); w=torch.softmax(self.time_attn(z).squeeze(-1),dim=1)
        return self.predict((z*w.unsqueeze(-1)).sum(1))
    def architecture_metadata(self):
        return {
            "implementation": type(self).__name__,
            "architecture_version": self.architecture_version,
            "source": "models.benchmark",
            "reproduction_scope": "architecture-level reproduction, not official authors' code",
            "input": "univariate DI prefix resampled to 128 steps",
            "order_encoding": "fixed sinusoidal positional encoding",
            "dual_layer_attention": [
                {"scope": "local", "local_radius": 8, "attention_heads": 4},
                {"scope": "global", "local_radius": 4, "landmark_stride": 16,
                 "attention_heads": 4},
            ],
            "decoder": "removed",
            "temporal_model": "one-layer bidirectional LSTM, 48 units per direction",
            "pooling": "learned temporal attention pooling",
            "output": "sigmoid-normalized scalar RUL",
        }


class DSResidual(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(channels,channels,3,padding=dilation,dilation=dilation,groups=channels),nn.Conv1d(channels,channels,1),nn.BatchNorm1d(channels),nn.ReLU())
    def forward(self,x): return x+self.net(x)


class AttnResDSCNN(Regressor):
    def __init__(self):
        super().__init__(); self.input_attn=nn.Sequential(nn.Conv1d(1,16,1),nn.Sigmoid()); self.stem=nn.Conv1d(1,64,5,padding=2); self.blocks=nn.Sequential(DSResidual(64,1),DSResidual(64,2),DSResidual(64,4)); self.pool=nn.AdaptiveAvgPool1d(1); self.head=nn.Linear(64,1)
    def forward(self,x):
        raw=x.unsqueeze(1); weight=self.input_attn(raw).mean(1,keepdim=True); z=torch.relu(self.stem(raw*weight)); return self.predict(self.pool(self.blocks(z)).squeeze(-1))


class DAGNetwork(Regressor):
    """DAG-style parallel CNN/LSTM feature graph and fusion node."""
    def __init__(self):
        super().__init__(); self.cnn=nn.Sequential(nn.Conv1d(1,32,5,padding=2),nn.ReLU(),nn.Conv1d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1)); self.lstm=nn.LSTM(1,64,2,batch_first=True,dropout=.1); self.fuse=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(.1)); self.head=nn.Linear(64,1)
    def forward(self,x):
        local=self.cnn(x.unsqueeze(1)).squeeze(-1); sequential,_=self.lstm(x.unsqueeze(-1)); return self.predict(self.fuse(torch.cat([local,sequential[:,-1]],-1)))


def sparse_mask(length,mode,device):
    index=torch.arange(length,device=device); distance=(index[:,None]-index[None,:]).abs(); allowed=distance<=4
    if mode=="prob": allowed=allowed | ((index[None,:] % max(1,length//16))==0)
    else:
        for power in (1,2,4,8,16,32,64): allowed=allowed | (distance==power)
    return ~allowed


class SparseBranch(nn.Module):
    def __init__(self,mode): super().__init__(); self.mode=mode; self.attn=nn.MultiheadAttention(64,4,batch_first=True); self.norm=nn.LayerNorm(64)
    def forward(self,x): y,_=self.attn(x,x,x,attn_mask=sparse_mask(x.size(1),self.mode,x.device)); return self.norm(x+y)


class IMDSSN(Regressor):
    """Deterministically masked ProbSparse/LogSparse dual-branch surrogate."""
    def __init__(self):
        super().__init__(); self.embed=nn.Linear(1,64); self.prob=SparseBranch("prob"); self.log=SparseBranch("log"); self.fuse=nn.Sequential(nn.Linear(128,64),nn.ReLU()); self.head=nn.Linear(64,1)
    def forward(self,x):
        z=self.embed(x.unsqueeze(-1)); return self.predict(self.fuse(torch.cat([self.prob(z).mean(1),self.log(z).mean(1)],-1)))


class PM2FN(Regressor):
    def __init__(self):
        super().__init__(); self.local=nn.Sequential(nn.Conv1d(1,32,3,padding=1),nn.ReLU(),nn.Conv1d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1)); self.global_branch=nn.Sequential(nn.Conv1d(1,32,9,padding=8,dilation=2),nn.ReLU(),nn.Conv1d(32,64,5,padding=8,dilation=4),nn.ReLU(),nn.AdaptiveAvgPool1d(1)); self.fuse=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(.1)); self.head=nn.Linear(64,1)
    def forward(self,x): return self.predict(self.fuse(torch.cat([self.local(x.unsqueeze(1)).squeeze(-1),self.global_branch(x.unsqueeze(1)).squeeze(-1)],-1)))


class KGHM(Regressor):
    """Data-driven two-node DWC/EDC relation-graph surrogate plus GRU."""
    def __init__(self):
        super().__init__(); self.node=nn.Linear(1,32); self.graph=nn.Linear(32,32); self.gru=nn.GRU(64,48,batch_first=True,bidirectional=True); self.head=nn.Linear(96,1)
    def forward(self,signals):
        # signals: B,T,2. Correlations are computed per batch using only input windows.
        centered=signals-signals.mean(1,keepdim=True); covariance=torch.einsum("bti,btj->bij",centered,centered)/(signals.size(1)-1); scale=torch.sqrt(torch.diagonal(covariance,dim1=1,dim2=2).clamp_min(1e-6)); adjacency=covariance/(scale[:,:,None]*scale[:,None,:]); adjacency=torch.softmax(adjacency,dim=-1)
        nodes=torch.relu(self.node(signals.unsqueeze(-1))); informed=torch.relu(self.graph(torch.einsum("bij,btjh->btih",adjacency,nodes))); sequence=informed.flatten(2); output,_=self.gru(sequence); return self.predict(output[:,-1])


def build_model(name):
    table={"CNN":CNN,"LSTM":lambda:RNNModel(False),"BiLSTM":lambda:RNNModel(True),"DAG_network":DAGNetwork,"IMDSSN":IMDSSN,"PM2FN":PM2FN,"Transformer":TransformerModel,"TCN_BiGRU":TCNBiGRU,"KGHM":KGHM,"ATCN":ATCN,"MDFA":MDFA,"SBi_Transformer":SBiTransformer,"Attn_ResDSCNN":AttnResDSCNN}
    if name not in table: raise KeyError("No runnable implementation for %s"%name)
    return table[name]()
