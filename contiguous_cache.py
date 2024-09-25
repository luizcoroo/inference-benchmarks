import torch

from llama import ModelArgs


class ContiguousKVCacheView:
    def __init__(self, kv_cache, start_pos, last_seq_len):
        self.k = kv_cache[0]
        self.v = kv_cache[1]
        self.start_pos = start_pos
        self.last_seq_len = last_seq_len

    def update_kv(self, xk, xv):
        bsz, seqlen, _, _ = xk.shape
        end_pos = self.start_pos + seqlen
        self.last_seq_len[0] = seqlen

        self.k[:bsz, self.start_pos : end_pos] = xk
        self.v[:bsz, self.start_pos : end_pos] = xv
        return (self.k[:bsz, :end_pos], self.v[:bsz, :end_pos])


class ContiguousKVCache:
    def __init__(self, args: ModelArgs):
        self.start_pos = 0
        self.last_seqlen = [0]
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

    def get_layer(self, layer_id: int):
        return ContiguousKVCacheView(
            self.kv_cache[layer_id],
            self.start_pos,
            self.last_seqlen,
        )

    def update(self):
        self.start_pos += self.last_seqlen[0]
