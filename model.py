import torch
import torch.nn as nn
import torch.optim as optim

from torch import Tensor
import torch.nn.functional as F

import numpy as np

def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    if mask is not None:
        temp += (mask * -1e9)
    softmax = F.softmax(temp / scale, dim=-1)
    return softmax.bmm(value), softmax  # Return attention scores as well as output

class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        output, attn = scaled_dot_product_attention(self.q(query), self.k(key), self.v(value), mask)
        return output, attn  # Return attention scores as well as output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        outputs, attns = zip(*(h(query, key, value, mask) for h in self.heads))
        return self.linear(torch.cat(outputs, dim=-1)), attns  # Return attention scores as well as output

#Age information in position_encoding
def position_encoding(
    seq_len: int, dim_model: int, age, device: torch.device
) -> Tensor:
    age = age.unsqueeze(-1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = age / (1e4 ** (dim / float(dim_model)))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )

class Residual1(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        if mask is not None:
            output, attention = self.sublayer(query, key, value, mask)
        else:
            output, attention = self.sublayer(query, key, value)
        return self.norm(query + self.dropout(output)), attention
        
class Residual2(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual1(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual2(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        src, attn = self.attention(src, src, src, mask = src_mask)
        return self.feed_forward(src), attn


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        device: torch.device = torch.device('cpu')
    ):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: Tensor, src_mask: Tensor, age: Tensor) -> Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        src += position_encoding(seq_len, dimension, age, self.device)
        attns = []
        for layer in self.layers:
            src, attn = layer(src, src_mask)
            attns.append(attn)

        return src, attns

class TransformerModel(nn.Module):
    def __init__(self, num_unique_codes, embedding_dim,  hidden_size, num_heads, num_encoder_layers, output_dim, pre_trained_weights=None, dim_feedforward=512, dropout: float = 0.1, device: torch.device = torch.device('cpu')):
#     def __init__(self, num_unique_codes, embedding_dim,  num_heads, num_encoder_layers, output_dim, dim_feedforward=512, dropout: float = 0.1, device: torch.device = torch.device('cpu')
# ):
        super().__init__()
        
        self.embedding = nn.Embedding(num_unique_codes, embedding_dim)

        if pre_trained_weights is not None:
            self.embedding.load_state_dict({'weight': pre_trained_weights})
            self.embedding.weight.requires_grad = True  # This line is optional, it makes embeddings non-trainable
            
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            dim_model=embedding_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            device = device
        )
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, src: Tensor, age: Tensor) -> Tensor:
        src = src.long() #[16,212] = [batch, seq_len]
        embedded_src = self.embedding(src) #[16,212,32] = [batch, seq_len, dim]
        src_mask = (src == 0).unsqueeze(1) #[batch, 1, seq_len]
        memory, attns = self.encoder(embedded_src, src_mask, age) # memomry = [batch, seq_len, dim], len(attn[0][0]) = 16
        
        mask4mem = ~src_mask.permute(0,2,1) #[batch, seq_len, 1]
        mask4mem = mask4mem.expand_as(memory) #[batch, seq_len, dim]
        
        # tensor = memory*mask4mem
        
        #weighted sum of memory:
        tensor = torch.bmm(attns[0][0], memory)*mask4mem
        
        sum_tensor = torch.sum(tensor, dim=1)
        num_elements = torch.sum(mask4mem, dim=1)
        mean_memory = sum_tensor/num_elements.clamp(min=1)
        
        x = self.fc1(mean_memory)
        x = self.relu(x)
        output = self.fc2(x)
        
        return output