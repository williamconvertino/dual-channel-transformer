import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_components import TransformerBlock, DualBlock

class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        if config.n_dual_blocks > 0:
            self.embedding = nn.Embedding(config.vocab_size, config.d_primary)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.d_latent)

        self.dual_blocks = nn.ModuleList(DualBlock(config, i) for i in range(config.n_dual_blocks))
        self.transformer_blocks = nn.ModuleList(TransformerBlock(config, i) for i in range(config.n_transformer_blocks))
        
        if config.n_dual_blocks > 0:
            self.ln_f = nn.LayerNorm(config.d_primary)
            self.lm_head = nn.Linear(config.d_primary, config.vocab_size, bias=False)
        else:
            self.ln_f = nn.LayerNorm(config.d_latent)
            self.lm_head = nn.Linear(config.d_latent, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.embedding.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
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
        
    def forward(self, x, targets=None, ignore_index=-1):
        
        B, S = x.shape

        x = self.embedding(x)
        
        if self.config.n_dual_blocks > 0:
            primary = secondary = x
            for dual_block in self.dual_blocks:
                primary, secondary = dual_block(primary, secondary)
            
        if self.config.n_dual_blocks > 0 and self.config.concat_streams_for_transformer:
            x = torch.cat((primary, secondary), dim=-1)
        elif self.config.n_dual_blocks > 0:
            x = primary
        else:
            x = x.view(B, S, -1)
            
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        if self.config.n_dual_blocks > 0:
            x = x[:, :, :self.config.d_primary]
        
        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        
        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=ignore_index)
        return logits, loss