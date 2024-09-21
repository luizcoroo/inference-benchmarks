import torch

import llama

vocab_size = 128256
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
    )
)


x = torch.randint(0, vocab_size, (1, 4096))
y = model(x, 0)
print(y.shape)
