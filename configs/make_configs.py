import numpy as np
import json

for n in range(1, 10):
    p = n / 10
    
    config = {
        "model": "dual",
        "max_seq_len": 128,
        "d_latent": 512,
        "p_primary": p,
        "n_dual_blocks": 2,
        "n_transformer_blocks": 1
    }
    
    with open(f"p_{n * 10}.json", "w") as f:
        json.dump(config, f, indent=4)