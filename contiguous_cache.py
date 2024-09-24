import torch

from llama import ModelArgs


class ContiguousKVCacheView:
    def __init__(self, kv_cache, start_pos):
        self.kv_cache = kv_cache
        self.start_pos = start_pos

    def update_kv(self, xk, xv):
        bsz, seqlen, _, _ = xk.shape
        self.kv_cache[0, :bsz, self.start_pos : self.start_pos + seqlen] = xk
        self.kv_cache[1, :bsz, self.start_pos : self.start_pos + seqlen] = xv
        return (
            self.kv_cache[0, :bsz, : self.start_pos + seqlen],
            self.kv_cache[1, :bsz, : self.start_pos + seqlen],
        )


class ContiguousKVCache:
    def __init__(self, args: ModelArgs):
        self.kv_cache = torch.zeros(
            (
                args.n_layers,
                2,
                args.max_batch_size,
                args.max_seq_len,
                args.n_heads if args.n_kv_heads is None else args.n_kv_heads,
                args.dim // args.n_heads,
            )
        )

    def get_layer(self, layer_id: int, start_pos: int):
        return ContiguousKVCacheView(self.kv_cache[layer_id], start_pos)
