import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_util import TransformerBlock, DualBlock, init_weights

class ShareEmbedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.primary_embedding = nn.Embedding(config.vocab_size, config.d_primary)
        
        self.dual_blocks = nn.ModuleList([
            DualBlock(config) for _ in range(config.n_dual_blocks)
        ])
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_transformer_blocks)
        ])
        
        self.ln_out = nn.LayerNorm(config.d_primary)
        self.lm_head = nn.Linear(config.d_primary, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.primary_embedding.weight
        
        self.apply(init_weights)

    def forward(self, x, targets=None, ignore_index=-1):
        
        B, S = x.shape
        device = x.device
        
        primary = self.primary_embedding(x)
        secondary = self.primary_embedding(x)
        
        for dual_block in self.dual_blocks:
            primary, secondary = dual_block(primary, secondary)
        
        if self.config.n_transformer_blocks > 0:
            
            x = torch.cat((primary, torch.zeros_like(secondary)), dim=-1)
            
            for transformer_block in self.transformer_blocks:
                x = transformer_block(x)
                
            primary = x[:, :, :self.config.d_primary]
            
        primary = self.ln_out(primary)
        
        logits = self.lm_head(primary)
        
        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=ignore_index)
        return logits, loss