import torch
import triton
from flash_attn import flash_attn_func


def exact_attn(query, key, value):
    return flash_attn_func(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2))

def sliding_window_attn(query, key, value, window_ratio=1/32):
    n = query.shape[2]
    window_size = int(window_ratio * window_ratio)
    return flash_attn_func(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2), window_size=(ws, ws))

def hyper_attn():
    pass


# b = 1
# h = 48
# n = 2**16
# d = 64
# device = 'cuda:0'
# dtype = torch.bfloat16

# query = torch.randn(b, h, n, d, device=device, dtype=dtype)
# key = torch.randn(b, h, n, d, device=device, dtype=dtype)
# value = torch.randn(b, h, n, d, device=device, dtype=dtype)

# func1 = lambda: flash_attn_func(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2))
# out = triton.testing.do_bench(func1, warmup=2_000, rep=10_000, quantiles=(0.2, 0.5, 0.8), return_mode='median')
# print(out)

# ws = int(n * (1/64))
# func2 = lambda: flash_attn_func(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2), window_size=(ws, ws))
# out2 = triton.testing.do_bench(func2, warmup=2_000, rep=10_000, quantiles=(0.2, 0.5, 0.8), return_mode='median')
# print(out2)

import pdb; pdb.set_trace();