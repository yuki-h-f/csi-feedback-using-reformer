import torch
import torch.nn.functional as F

TOKEN_SELF_ATTN_VALUE = -5e4 # carefully set for half precision to work
def sort_key_val(t1: torch.Tensor, t2: torch.Tensor, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)

def batched_index_select(values:torch.Tensor, indices: torch.Tensor):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

class LshAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_hashes = 8


    def hash_vectors(self, x: torch.Tensor, n_buckets: int) -> torch.Tensor:
        batch_size = x.shape[0]
        rotation_size = n_buckets
        rotation_shape = (
            batch_size,
            x.shape[-1],
            self.n_hashes,
            rotation_size //2
        )
        random_rotations = torch.randn(rotation_shape, dtype=x.dtype, device=x.device).expand(batch_size, -1, -1, -1)
        rotated_x = torch.einsum('btf, bfhi->bhti', x, random_rotations)
        rotated_x = torch.cat([rotated_x, -rotated_x], dim=1)
        buckets = torch.argmax(rotated_x, -1)
        offsets = torch.arange(self.n_hashes, device=x.device)
        offsets = torch.reshape(offsets * n_buckets, (1, -1, -1))
        buckets = torch.reshape(buckets + offsets, (batch_size, -1,))
        return buckets
    
    def forward(self, qk: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch_size, seqlen, dim, device = *qk.shape, qk.device
        bucket_size = 16
        n_buckets = seqlen // bucket_size
        buckets = self.hash_vectors(qk, n_buckets)
        total_hashes = self.n_hashes
        ticker = torch.arange(total_hashes * seqlen, device=device).unsqueeze(0).expand_as(buckets)
        buckets_and_ticker = seqlen + buckets + (ticker % seqlen)
        buckets_and_ticker = buckets_and_ticker.detach()
        
        sbuckets_and_t, sticker = sort_key_val(buckets_and_ticker, ticker, dim=-1)
        _, undo_sort = sticker.sort(dim=-1)
        del ticker

        sbuckets_and_t = sbuckets_and_t.detach()
        sticker = sticker.detach()
        undo_sort = undo_sort.detach()

        st = (sticker % seqlen)
        sqk = batched_index_select(qk, st)
        sv = batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        chunk_size = total_hashes * n_buckets
        bq_t = bkv_t = torch.reshape(st, (batch_size, chunk_size, -1))
        bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))
        bv = torch.reshape(sv, (batch_size, chunk_size, -1, dim))

        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk
        bk = F.normalize(bqk, p=2, dim=-1).type_as(bq)

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x):
            x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
            return torch.cat([x, x_extra], dim=2)

        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)

        # Dot-product attention.
        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (dim ** -0.5)
        masked_value = max_neg_value(dots)


        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
        del self_mask

        # Softmax.
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp).type_as(dots)
        dropped_dots = self.dropout(dots)

        bo = torch.einsum('buij,buje->buie', dropped_dots, bv)
        so = torch.reshape(bo, (batch_size, -1, dim))
        slogits = torch.reshape(dots_logsumexp, (batch_size, -1,))

        # unsort logits
        o = batched_index_select(so, undo_sort)
        logits = slogits.gather(1, undo_sort)

        o = torch.reshape(o, (batch_size, total_hashes, seqlen, dim))
        logits = torch.reshape(logits, (batch_size, total_hashes, seqlen, 1))

        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        out = torch.sum(o * probs, dim=1)

        attn = torch.empty(0, device=device)
        return out, attn, buckets
    


