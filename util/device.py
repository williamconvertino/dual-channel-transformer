import torch

def get_device():
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu')
    
    props = torch.cuda.get_device_properties(0)
    gpu = torch.device(f'cuda:{0}')
    free_memory, total_memory = torch.cuda.mem_get_info(gpu)
    total_memory = int(total_memory / 1024**3)
    free_memory = int(free_memory / 1024**3)  
    print(f"Using GPU 0: {props.name} with {free_memory:.2f}GB")
    
    return gpu    