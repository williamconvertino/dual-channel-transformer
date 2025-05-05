import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_util import ReducedScratchSpaceBlock, init_weights

class ReducedScratchSpaceModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.config.d_reduced = config.d_latent - int(config.d_latent * config.p_reduced) 
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_reduced)
        
        self.transformer_blocks = nn.ModuleList([
            ReducedScratchSpaceBlock(config) for _ in range(config.n_layers)
        ])
        
        self.ln_out = nn.LayerNorm(config.d_reduced)
        self.lm_head = nn.Linear(config.d_reduced, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.embedding.weight
        
        self.apply(init_weights)

    def forward(self, x, targets=None, ignore_index=-1):
        
        B, S = x.shape
        
        x = self.embedding(x)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        x = x[:, :, :self.config.d_reduced] # Remove padding
        
        x = self.ln_out(x)
        
        logits = self.lm_head(x)
        
        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=ignore_index)
        return logits, loss