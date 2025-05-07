import os

model_dirs = os.listdir("checkpoints")

for model_dir in model_dirs:
    checkpoint_files = os.listdir(os.path.join("checkpoints", model_dir))
    checkpoint_files = [f for f in checkpoint_files if f.endswith(".pt") and f.startswith("epoch_")]
    checkpoint_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)
    latest_checkpoint = checkpoint_files[0] if checkpoint_files else None
    
    print(f"Model: {model_dir} | Epoch: {latest_checkpoint.split('_')[1].split('.')[0] if latest_checkpoint else 'N/A'}")