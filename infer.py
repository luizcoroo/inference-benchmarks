import torch

import llama

vocab_size = 128256
batch_size = 4
min_tokens = 256
max_tokens = 512

print("Loading model")
model = llama.Transformer(
    llama.ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=vocab_size,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        norm_eps=1e-05,
        rope_theta=500000.0,
        max_batch_size=batch_size,
        max_seq_len=max_tokens,
    ),
    device="cuda",
    dtype=torch.bfloat16,
)

print("executing model")
model.eval()

x = torch.randint(0, vocab_size, (batch_size, min_tokens), device="cuda")

start_pos = 0
for end_pos in range(min_tokens, max_tokens):
    y = model(x[:, start_pos:end_pos], start_pos)
    x = torch.cat((x, torch.argmax(y[:, -1, :], dim=-1, keepdim=True)), 1)
    start_pos = end_pos

print(x.shape)
