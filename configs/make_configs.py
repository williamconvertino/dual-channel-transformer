import json

for n in range(1, 10):
    p = n / 10
    
    re_config = {
        "model_type": "reduced_embedding",
        "p_reduced": p,
        "d_latent": 512,
        "max_seq_len": 128,
        "n_heads": 8,
        "n_layers": 3
    }
    
    with open(f"r_embed_{n * 10}.json", "w") as f:
        json.dump(re_config, f, indent=4)
        
    
    rss_config = {
        "model_type": "reduced_scratch_space",
        "p_reduced": p,
        "d_latent": 512,
        "max_seq_len": 128,
        "n_heads": 8,
        "n_layers": 3
    }
    
    with open(f"r_ss_{n * 10}.json", "w") as f:
        json.dump(rss_config, f, indent=4)