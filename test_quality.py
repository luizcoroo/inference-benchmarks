import os
import random

import numpy as np
import torch
from tqdm import tqdm

import llama
from contiguous_cache import ContiguousKVCache

np.random.seed(333)
random.seed(333)
torch.manual_seed(333)
torch.cuda.manual_seed_all(333)
torch.set_default_device("cuda")


dtypes = [torch.float32, torch.float16]
seq_lens = [256]
batch_sizes = [1, 64]
use_kv_caches = [False, True]
n_examples = 100
llama_path = "llama-8B/original/consolidated.00.pt"

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
)


def load_model(params, dtype):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    model = llama.Transformer(params)
    model.load_state_dict(torch.load(llama_path, map_location="cpu", weights_only=True))
    torch.set_default_dtype(old_dtype)
    model.eval()
    return model


def enlarge_batch_to(x, batch_size, seq_len, vocab_size):
    if batch_size == 1:
        return x

    return torch.cat((x, torch.randint(0, vocab_size, (batch_size - 1, seq_len))))


def get_hidden_states(model, batch, dtype, use_kv_cache=False):
    if not use_kv_cache:
        hidden_states = model(batch, return_hidden_states=True)["hidden_states"]
    else:
        bsz, seqlen = batch.shape
        cache = ContiguousKVCache(params, bsz, seqlen, dtype)
        logits = model(batch[:, :-1], cache)["logits"]
        next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        cache.update()
        hidden_states = model(next, cache, return_hidden_states=True)["hidden_states"]

    return [h[0, -1, :] for h in hidden_states]


def prepare_ground_truth(seq_lens, vocab_size, n_examples):
    dtype = torch.float32
    model = load_model(params, dtype)
    xs, hs = [], []
    for seq_len in seq_lens:
        xs.append([])
        hs.append([])
        for _ in range(n_examples):
            xs[-1].append(torch.randint(0, vocab_size, (1, seq_len)))
            hs[-1].append(get_hidden_states(model, xs[-1][-1], dtype, False))

    return xs, hs


def compute_metrics(hs, hidden_states):
    metrics = {"rmse": [], "mad": []}
    for hidden_state1, hidden_state2 in zip(hs, hidden_states):
        diff = hidden_state1 - hidden_state2
        metrics["rmse"].append(torch.sqrt(torch.sum(diff**2) / len(diff)).item())
        metrics["mad"].append(torch.max(torch.abs(diff)).item())

    return metrics


print("computing ground truth for sequence lengths", seq_lens)
xs, hs = prepare_ground_truth(seq_lens, params.vocab_size, n_examples)

print(f"parameters: {dtypes=}, {seq_lens=}, {batch_sizes=}, and {use_kv_caches=}")
configs = []
for dtype in dtypes:
    for input_id, _ in enumerate(xs):
        for batch_size in batch_sizes:
            for use_kv_cache in use_kv_caches:
                configs.append((dtype, input_id, batch_size, use_kv_cache))


last_dtype = ""
vocab_size = params.vocab_size
with open("quality.csv", "w") as file:
    for dtype, input_id, batch_size, use_kv_cache in tqdm(configs):
        if dtype != last_dtype:
            model = load_model(params, dtype)
        seq_len = xs[input_id][0].shape[1]
        metrics = []
        for x, h in zip(xs[input_id], hs[input_id]):
            assert x.shape[1] == seq_len
            batch = enlarge_batch_to(x, batch_size, seq_len, vocab_size)
            hidden_states = get_hidden_states(model, batch, dtype, use_kv_cache)
            assert len(h) == len(hidden_states)
            metrics.append(compute_metrics(h, hidden_states))

        label = f"{dtype},{seq_len},{batch_size},{use_kv_cache}"
        compute_mean = lambda x: np.mean([m[x] for m in metrics], axis=0)
        file.write(label + "," + f'"{compute_mean("rmse")}","{compute_mean("mad")}"')
        file.write("\n")
        last_dtype = dtype
