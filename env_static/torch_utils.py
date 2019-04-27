import numpy as np
import torch
import torch.nn as nn

def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

"""
GPU wrappers from 
https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/pytorch_util.py
"""

_use_gpu = False
device = None
device_str = ''

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu, device, _gpu_id, device_str
    _gpu_id = gpu_id
    _use_gpu = mode
    device_str = "cuda:" + str(gpu_id) if _use_gpu else "cpu"
    device = torch.device(device_str)

def gpu_enabled():
    return _use_gpu

def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    kwargs['devic'] = device_str
    return torch.FloatTensor(*args, **kwargs)

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def get_numpy(tensor):
    # not sure if I should do detach or not here
    return tensor.to('cpu').detach().numpy()

def zeros(*sizes, **kwargs):
    kwargs['device'] = device_str
    return torch.zeros(*sizes, **kwargs)

def ones(*sizes, **kwargs):
    kwargs['device'] = device_str
    return torch.ones(*sizes, **kwargs)

def randn(*args, **kwargs):
    kwargs['device'] = device_str
    return torch.randn(*args, **kwargs)

def zeros_like(*args, **kwargs):
    kwargs['device'] = device_str
    return torch.zeros_like(*args, **kwargs)

def normal(*args, **kwargs):
    kwargs['device'] = device_str
    return torch.normal(*args, **kwargs)

def eye(*args, **kwargs):
    kwargs['device'] = device_str
    return torch.eye(*args, **kwargs)

def batch_diag(A):
    n = A.size(-1)
    B = eye(n)
    C = A.unsqueeze(len(A.size())).expand(*A.size(), n)

    return C * B

def batch_apply(f, M):
    tList = [f(m) for m in torch.unbind(M, dim=0)]
    res = torch.stack(tList, dim=0)

    return res 