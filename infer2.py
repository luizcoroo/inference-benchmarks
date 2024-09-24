import os
import random

import numpy as np
import torch
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from tqdm import tqdm

import llama_raw as llama

model_parallel_size = None

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group("nccl")
if not model_parallel_is_initialized():
    if model_parallel_size is None:
        model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
    initialize_model_parallel(model_parallel_size)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

torch.set_default_tensor_type(torch.cuda.FloatTensor)
# torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)

np.random.seed(333)
random.seed(333)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
torch.manual_seed(333)
torch.cuda.manual_seed_all(333)
# torch.set_printoptions(precision=10)


max_batch_size = 4
min_seq_len = 32
max_seq_len = 128
max_iterations = 50


print("Initializing model")


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
    max_batch_size=max_batch_size,
    max_seq_len=max_seq_len,
)


model = llama.Transformer(params)
model.load_state_dict(
    torch.load(
        "/cortex/models/base/llama-3.1-8b/consolidated.00.pth",
        map_location="cpu",
    ),
    strict=False,
)


# oi ola como posso te ajudar
## static batching until context is full
print("executing model")
model.eval()

score = 0
for _ in tqdm(range(max_iterations)):
    x1 = torch.randint(
        0,
        params.vocab_size,
        (max_batch_size, min_seq_len),
    )

    x2 = x1.clone()

    start_pos = 0
    for end_pos in tqdm(range(min_seq_len, max_seq_len), leave=False):
        y = model(x1[:, start_pos:end_pos], start_pos)
        x1 = torch.cat((x1, torch.argmax(y[:, -1, :], dim=-1, keepdim=True)), 1)

    for end_pos in tqdm(range(min_seq_len, max_seq_len), leave=False):
        y = model(x2[:, start_pos:end_pos], start_pos)
        x2 = torch.cat((x2, torch.argmax(y[:, -1, :], dim=-1, keepdim=True)), 1)
        start_pos = end_pos

    differences = torch.tensor([max_seq_len] * max_batch_size)
    for i in range(0, max_batch_size):
        for j in range(0, max_seq_len):
            if x1[i, j] != x2[i, j]:
                differences[i] = j
                break

    current_score = torch.mean(
        (differences - min_seq_len) / (max_seq_len - min_seq_len)
    )
    score += current_score
    print(current_score)

print(f"Finished with total score = {score / max_iterations}")

# 0.8632812
# 0.8515625
