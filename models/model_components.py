import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

class Attention(nn.Module):
    def __init__(self, config, d_q, d_k, d_v, d_out):
        super().__init__()
        
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
    
class FeedForward(nn.Module):
    def __init__(self, config, d_in, d_out):
        super().__init__()

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
    def __init__(self, config, layer):
        super().__init__()
        
        self.config = config
        
        if config.n_dual_blocks > 0 and layer == 0:
            if config.concat_streams_for_transformer:
                d_q = config.d_primary + config.d_secondary
                d_k = config.d_primary + config.d_secondary
                d_v = config.d_primary + config.d_secondary

                self.ln_attn = nn.LayerNorm(config.d_primary + config.d_secondary)
            else:
                d_q = config.d_primary
                d_k = config.d_primary
                d_v = config.d_primary
                self.ln_attn = nn.LayerNorm(config.d_primary)
        else:
            d_q = config.d_latent
            d_k = config.d_latent
            d_v = config.d_latent
            self.ln_attn = nn.LayerNorm(config.d_latent)

        self.attention = Attention(config, d_q=d_q, d_k=d_k, d_v=d_v, d_out=config.d_latent)
        
        self.feed_forward = FeedForward(config, d_in=config.d_latent, d_out=config.d_latent)
        self.ln_ff = nn.LayerNorm(config.d_latent)
        
    def forward(self, x):
        x = x + self.attention(self.ln_attn(x))
        x = x + self.feed_forward(self.ln_ff(x))
        
        return x
    
class DualBlock(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        
        self.config = config
        
        self.attention = Attention(config, d_q=config.d_primary, d_k=config.d_primary, d_v=config.d_secondary, d_out=config.d_primary)
        
        if config.concat_streams_for_ff:
            self.feed_forward = FeedForward(config, d_in=config.d_primary + config.d_secondary, d_out=config.d_secondary)
        else:
            self.feed_forward = FeedForward(config, d_in=config.d_primary, d_out=config.d_secondary)
            
        self.ln_primary = nn.LayerNorm(config.d_primary)
        self.ln_secondary = nn.LayerNorm(config.d_secondary)
        
        if config.concat_streams_for_ff:
            self.ln_ff = nn.LayerNorm(config.d_primary + config.d_secondary)
        else:
            self.ln_ff = nn.LayerNorm(config.d_primary)
        
    def forward(self, primary, secondary):

        qk = self.ln_primary(primary)
        v = self.ln_secondary(secondary)

        primary = primary + self.attention(q=qk, k=qk, v=v)
        
        if self.config.concat_streams_for_ff:
            ff_in = torch.cat((primary, secondary), dim=-1)
        else:
            ff_in = primary
        
        ff_in = self.ln_ff(ff_in)
        ff_out = self.feed_forward(ff_in)
        
        if self.config.dual_ff_residual:
            secondary = secondary + ff_out
        else:
            secondary = ff_out
            
        return primary, secondary