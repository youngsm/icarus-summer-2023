import torch

__all__ = ['device', 'device_ids', 'map_location']

device = None
device_ids = None
map_location = None

def init_device():
    """
    checks for available devices and sets the following global variables:
    
    system | device | device_ids  | map_location
    -------|--------|-------------|-------------
    CUDA   | cuda:0 | [0,1,2,...] | None
    Metal  | mps    | None        | mps
    CPU    | cpu    | None        | cpu
    -------|--------|-------------|-------------
    
    you can specify the devices you want by explicitly
    setting backend.device, backend.device_ids, and backend.map_location.
    """
    
    global device, device_ids, map_location
    
    # CUDA-enabled gpus
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_ids = list(range(torch.cuda.device_count()))    
        
    # Mac Metal (M1, M2, etc.)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        map_location = device
        
    # CPU
    else:
        device = torch.device("cpu")
        map_location = device

init_device()