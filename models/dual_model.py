import torch.nn as nn
import torch.nn.functional as F
from .model_util import DualBlock, init_weights

class DualModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_latent)
        
        self.dual_blocks = nn.ModuleList([
            DualBlock(config) for _ in range(config.n_layers)
        ])
        
        self.ln_out = nn.LayerNorm(config.d_latent)
        self.lm_head = nn.Linear(config.d_latent, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.embedding.weight
        
        self.apply(init_weights)

    def forward(self, x, targets=None, ignore_index=-1):
        
        B, S = x.shape
        device = x.device
        
        primary = secondary = self.embedding(x)
        
        for dual_block in self.dual_blocks:
            primary, secondary = dual_block(primary, secondary)
        
        x = self.ln_out(secondary) # Use secondary to take advantage of final MLP
        logits = self.lm_head(x)
        
        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=ignore_index)
        return logits, loss