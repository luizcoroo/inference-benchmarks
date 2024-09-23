import torch

from llama import ModelArgs


class ContiguousKVCacheView:
    def __init__(self, kv_cache):
        self.k_cache = kv_cache[0]
        self.v_cache = kv_cache[1]

    def get_kv_cache(self):
        return self.k_cache, self.v_cache

class ContiguousKVCache:
    def __init__(
        self,
        params: ModelArgs,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        head_dim = params.dim // params.n_heads
        self.kv_cache = torch.zeros(
            (
                params.n_layers,
                2,
                params.max_batch_size,
                params.max_seq_len,
                params.n_kv_heads,
                head_dim,
            ),
            device=device,
            dtype=dtype,
        )

    def get_layer(self, layer_id: int):
        return ContiguousKVCacheView(self.kv_cache[layer_id])
