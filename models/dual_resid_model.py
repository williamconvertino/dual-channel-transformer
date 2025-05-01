import torch.nn as nn
import torch.nn.functional as F
from .model_util import DualResidBlock, init_weights

class DualResidModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_latent)
        
        self.dual_blocks = nn.ModuleList([
            DualResidBlock(config, layer) for layer in range(config.n_dual_blocks)
        ])
        
        self.ln_out = nn.LayerNorm(config.d_latent)
        self.lm_head = nn.Linear(config.d_latent, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.primary_embedding.weight
        
        self.apply(init_weights)

    def forward(self, x, targets=None, ignore_index=-1):
        
        B, S = x.shape
        device = x.device
        
        primary, secondary = self.embedding(x)
        
        for dual_block in self.dual_blocks:
            primary, secondary = dual_block(primary, secondary)
        
        primary = self.ln_out(primary)
        logits = self.lm_head(primary)
        
        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=ignore_index)
        return logits, loss