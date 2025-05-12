import json

for n in range(1, 12):
    
    if n == 10:
        p = 0.95
        name = "95"
    elif n == 11:
        p = 0.99
        name = "99"
    else:
        p = n / 10
        name = str(n * 10)
    
    re_config = {
        "model_type": "reduced_embedding",
        "p_reduced": p,
        "d_latent": 512,
        "max_seq_len": 128,
        "n_heads": 8,
        "n_layers": 3
    }
    
    with open(f"r_embed_{name}.json", "w") as f:
        json.dump(re_config, f, indent=4)
        
    rss_config = {
        "model_type": "reduced_scratch_space",
        "p_reduced": p,
        "d_latent": 512,
        "max_seq_len": 128,
        "n_heads": 8,
        "n_layers": 3
    }
    
    with open(f"r_ss_{name}.json", "w") as f:
        json.dump(rss_config, f, indent=4)
        
    st_config = {
        "model_type": "transformer",
        "p_reduced": p,
        "d_latent": 512,
        "max_seq_len": 128,
        "n_heads": 8,
        "n_layers": 3
    }
    
    
    with open(f"st_{name}.json", "w") as f:
        json.dump(st_config, f, indent=4)