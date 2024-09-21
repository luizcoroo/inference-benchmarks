import torch

import llama
from contiguous_cache import ContiguousKVCache

print("Loading model")
params = llama.ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    vocab_size=128256,
    ffn_dim_multiplier=1.3,
    multiple_of=1024,
    norm_eps=1e-05,
    rope_theta=500000.0,
    max_batch_size=4,
    max_seq_len=512,
)


model = llama.Transformer(
    params,
    device="cuda",
    dtype=torch.bfloat16,
)

cache = ContiguousKVCache(
    params,
    device="cuda",
    dtype=torch.bfloat16,
)

## static batching until context is full
print("executing model")
model.eval()

start_pos = 0
min_seq_len = 256
x = torch.randint(
    0,
    params.vocab_size,
    (params.max_batch_size, min_seq_len),
    device="cuda",
)

for end_pos in range(min_seq_len, params.max_seq_len):
    y = model(x[:, start_pos:end_pos], start_pos, cache)
    x = torch.cat((x, torch.argmax(y[:, -1, :], dim=-1, keepdim=True)), 1)
    start_pos = end_pos

print(x.shape)
