import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

def init_weights(module):
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
class Attention(nn.Module):
    def __init__(self, config, d_q=None, d_k=None, d_v=None, d_out=None):
        super().__init__()
        
        if d_q is None:
            d_q = config.d_latent
        if d_k is None:
            d_k = config.d_latent
        if d_v is None:
            d_v = config.d_latent
        if d_out is None:
            d_out = config.d_latent
        
        self.config = config
        
        self.W_q = nn.Linear(d_q, config.n_heads * config.d_latent, bias=False)
        self.W_k = nn.Linear(d_k, config.n_heads * config.d_latent, bias=False)
        self.W_v = nn.Linear(d_v, config.n_heads * config.d_latent, bias=False)
        self.W_o = nn.Linear(config.n_heads * config.d_latent, d_out, bias=False)
        
        self.attn_scale = 1 / math.sqrt(config.d_latent)
        
        self.rotary_embeddings = RotaryPositionalEmbeddings(config.d_latent)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
    def forward(self, q, k=None, v=None):
        
        if k is None:
            k = q
        if v is None:
            v = q
        
        B, S, E = q.shape
        
        q = self.W_q(q).view(B, S, self.config.n_heads, self.config.d_latent).transpose(1, 2)
        k = self.W_k(k).view(B, S, self.config.n_heads, self.config.d_latent).transpose(1, 2)
        v = self.W_v(v).view(B, S, self.config.n_heads, self.config.d_latent).transpose(1, 2)
        
        q = self.rotary_embeddings(q)
        k = self.rotary_embeddings(k)
        
        causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool().to(q.device)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.drop_attn(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.config.d_latent * self.config.n_heads)
        attn_output = self.W_o(attn_output)
        attn_output = self.drop_resid(attn_output)
        
        return attn_output
    
class MLP(nn.Module):
    def __init__(self, config, d_in=None, d_out=None):
        super().__init__()

        if d_in is None:
            d_in = config.d_latent
        if d_out is None:
            d_out = config.d_latent

        self.fc_1 = nn.Linear(d_in, 4 * config.d_latent)
        self.fc_2 = nn.Linear(4 * config.d_latent, d_out)
        
        self.activation = nn.GELU()    
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc_2(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.attention = Attention(config)
        self.ln_attn = nn.LayerNorm(config.d_latent)
        
        self.mlp = MLP(config)
        self.ln_mlp = nn.LayerNorm(config.d_latent)
        
    def forward(self, x):
        x = x + self.attention(self.ln_attn(x))
        x = x + self.mlp(self.ln_mlp(x))
        return x
    
class ReducedScratchSpaceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.attention = Attention(config, d_q=config.d_reduced, d_k=config.d_reduced, d_v=config.d_reduced, d_out=config.d_reduced)
        self.ln_attn = nn.LayerNorm(config.d_reduced)
        
        self.mlp = MLP(config, d_in=config.d_reduced, d_out=config.d_reduced)
        self.ln_mlp = nn.LayerNorm(config.d_reduced)
        
    def forward(self, x):
        x = x + self.attention(self.ln_attn(x))
        x = x + self.mlp(self.ln_mlp(x))
        return x

class DualBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.attention = Attention(config)
        self.ln_attn = nn.LayerNorm(config.d_latent)
        
        self.mlp = MLP(config)
        self.ln_mlp = nn.LayerNorm(config.d_latent)
            
    def forward(self, primary, secondary):

        primary = primary + self.attention(self.ln_attn(secondary))
        
        if self.config.mlp_residual == "secondary":
            secondary = secondary + self.mlp(self.ln_mlp(primary))
        else:
            secondary = primary + self.mlp(self.ln_mlp(primary)) # Default behavior
        
        return primary, secondary