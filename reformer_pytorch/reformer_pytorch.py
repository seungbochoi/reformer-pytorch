import math
import torch
import torch.nn as nn
from torch.nn import Identity
import torch.nn.functional as F
from torch.autograd import Function
from functools import partial, reduce, wraps
from itertools import chain
from operator import mul
from reformer_pytorch.reversible import ReversibleSequence

# constants

TOKEN_SELF_ATTN_VALUE = -5e4  # carefully set for half precision to work


# helper fns

def sort_key_val(t1, t2, dim=-1):
    # indices = sorted 가 어떻게됬는지 오리지널 오더 인덱스 로케이션도줌.
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


def process_inputs_chunk(fn, chunks=1, dim=0):
    def inner_fn(*args, **kwargs):
        keys, values, len_args = kwargs.keys(), kwargs.values(), len(args)
        chunked_args = list(zip(*map(lambda x: x.chunk(chunks, dim=dim), list(args) + list(values))))
        all_args = map(lambda x: (x[:len_args], dict(zip(keys, x[len_args:]))), chunked_args)
        outputs = [fn(*c_args, **c_kwargs) for c_args, c_kwargs in all_args]
        return tuple(map(lambda x: torch.cat(x, dim=dim), zip(*outputs)))

    return inner_fn


def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tensor.reshape(-1, last_dim)
    summed_tensors = [c.sum(dim=-1) for c in tensor.chunk(chunks, dim=0)]
    return torch.cat(summed_tensors, dim=0).reshape(orig_size)


def default(val, default_val):
    return default_val if val is None else val


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def cache_fn(f):
    cache = None

    @wraps(f)  # decorator + maintain the information of the cache target f(아규먼트값, 파라미터값 등등 전부 홀드 !)
    def cached_fn(*args, **kwargs):
        nonlocal cache  # 바로위 스코프에있는 cache!
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


def cache_method_decorator(cache_attr, cache_namespace, reexecute=False):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, key_namespace=None, fetch=False, set_cache=True, **kwargs):
            namespace_str = str(default(key_namespace, ''))
            _cache = getattr(self, cache_attr)
            _keyname = f'{cache_namespace}:{namespace_str}'

            if fetch:
                val = _cache[_keyname]
                if reexecute:
                    fn(self, *args, **kwargs)
            else:
                val = fn(self, *args, **kwargs)
                if set_cache:
                    setattr(self, cache_attr, {**_cache, **{_keyname: val}})
            return val

        return wrapper

    return inner_fn


def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)


def expand_dim(dim, k, t):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]


# helper classes

class MatrixMultiply(nn.Module):
    def __init__(self, tensor, transpose=False, normalize=False):
        super().__init__()
        self.tensor = tensor
        self.transpose = transpose
        self.normalize = normalize

    # 4,4096,512 ->
    # [4,12] ->
    def forward(self, x):
        tensor = self.tensor
        if self.normalize:
            tensor = F.normalize(tensor, dim=-1)
        if self.transpose:
            tensor = tensor.t()
            # x=4,4096,512 * tensor = token_emb(256,512)
        return x @ tensor  # 4,4096,256


class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(1))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1))  # learnable parameter 딱 하나 !
        self.eps = eps

    def forward(self, x):
        # L2 norm !

        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)

        return x / n * self.g # self.g 도 리턴해줘.

class ScaleNormMatt(nn.Module):
    def __init__(self, dim, g, eps=1e-5):
        super().__init__()
          # learnable parameter g 딱 하나 !
        self.g = g
        self.eps = eps

    def forward(self, x):
        # L2 norm !
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / n * self.g # self.g 도 리턴해줘.


# X2 = X1 + F(SclaeNorm(X1)) ????? 이게아니라 걍 x2 = F(scaleNorm(x1))

class PreNorm(nn.Module):
    def __init__(self, norm_class, dim, fn):
        super().__init__()
        self.norm = norm_class(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)  # fn = attn ! LSHSelfAttention
        # x,g = self.norm(x)
        # return self.fn(x, g, **kwargs)


class ChunkBeforeFF(nn.Module):
    def __init__(self, chunks, fn, along_dim=-1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        # x=4,4096,512 -> 4,410,512 10개로 나
        chunks = x.chunk(self.chunks, dim=self.dim) # x인풋을 짤라서 느겠다 ..

        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim=self.dim)


# LSH attention as described in https://openreview.net/pdf?id=rkgNKkHtvB
# adapted from trax, stripped to what paper said needed to work
# namely that buckets need to be at least 64 with 8 rounds of hashing
# https://github.com/google/trax/blob/master/trax/layers/research/efficient_attention.py#L442

class LSHAttention(nn.Module):
    def __init__(self,
                 dropout=0.,
                 bucket_size=64,
                 n_hashes=8,
                 causal=False,
                 allow_duplicate_attention=True,
                 attend_across_buckets=True,
                 rehash_each_round=True,
                 drop_for_hash_rate=0.0,
                 random_rotations_per_head=False,
                 return_attn=False):
        super().__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = nn.Dropout(dropout)
        self.dropout_for_hash = nn.Dropout(drop_for_hash_rate)

        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.bucket_size = bucket_size

        self.n_hashes = n_hashes

        self._allow_duplicate_attention = allow_duplicate_attention  # ????
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head

        # will expend extra computation to return attention matrix
        self._return_attn = return_attn

        # cache buckets for reversible network, reported by authors to make Reformer work at depth
        self._cache = {}

    # 이 데코레이터머임??
    @cache_method_decorator('_cache', 'buckets', reexecute=True)
    def hash_vectors(self, n_buckets, vecs):
        batch_size = vecs.shape[0]
        device = vecs.device

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        assert n_buckets % 2 == 0  # ? 왜 이래야함?

        rot_size = n_buckets # 64 <-색갈 갯수(혹은 구역 그림에서 원의)

        # 1 64 4 32
        rotations_shape = (
            batch_size if self._random_rotations_per_head else 1,  # 1
            vecs.shape[-1],  # 64
            self.n_hashes if self._rehash_each_round else 1,  # 4
            rot_size // 2)  # 64//2
        # 16, 64, 4, 32로 늘어남 ??
        # 얘가 R이다 xR의 R

        random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype, device=device) \
            .expand(batch_size, -1, -1, -1)
            # 1,64,4,32 -> b:16, f:64, h:4, i:32 // 즉 16개로 tiling(ㅎㅎ 샬로카피랑 비슷) 함
            # shape interpretation:
            # 64 = 4096캐릭터를 reprep하는 vector의 각 캐릭터의 디멘션(512그부분) paper의 b pr 버켓색갈 갯수(bucket_size)
            # 4 = N round hash
            # 32 = 저 64를 32로 줄인다 ? paper이해못해서 몰라 ㅋㅋ
            #16번 똑같은 R로 xR해주겠다 !

        # 4,4096,512 -> 멀티헤드갯수(4) 만큼 똑같은짓(self-attn)을 다른 learnable parameter로 해준다
        # qk 한개로 4 개의 INDEPEndent 하게 self-attn 배움
        # 4,4096,512 -> 4,4,4096,512 가 됌(batch를 고려하면)
        #           -> 여기서 merge_batch_and_heads써서 16,4096,512 로 합쳐서 계산진행(나중에 결과값은 view 해주면되니까)
        #
        #   self-attn0   -> 얘네끼리도(4개 다른 flow) 어텐션을 배운다 누구 플로가 더 중요한지.
        #   self-attn1
        #   self-attn2
        #   self-attn3
        # 원래 vec은 4,4096,512였는데 4,4,4096,64로 어디선가 바꿔주고(아직안봄)
        # multi-head부분에서 merge_batch_and_heads 여기서 16,4096,512(이해못함아직)
        dropped_vecs = self.dropout_for_hash(vecs)
        # 4096개의 64 dim인 벡터가 각각있고, 각각을 random projection하고싶다. R(16,64,4,32)을 써서
        #       ..(그리고 xR;-xR컨캣해주면 그게 로테이션+clustering의 효과가있다고 하는거야 페이퍼에서)
        # 우리는 64D 에있는거야
        # qk = 16,4096,512(4,4,4096,512)
        # 16,4096,64 -> 16,4,4096,32 로 변함..
        # 16,4096,64 * R(16,64,4,32)
        #   n_hashround(4)무시하고, multihead batch size(16), 캐릭터갯수(4096) 무시하면
        #   결국 난 각 캐릭터 디멘션(reprep_dim)[64] 를 프로젝션 다른사이즈로(32) 하고싶은거야.
        #   결국 이건 64 * R(64,32) 이거랑같아
        #  b    t  f      b  f h  i
        #       t              h  i
        #                              f  f
        rotated_vecs = torch.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)
        # rotated_vecs_study = torch.einsum('btf,bfhi->bith', dropped_vecs, random_rotations)
        # 이부분이 LSH h(x) 구하는부분
        if self._rehash_each_round:
            # 4번 해싱(돌리는거함)
            # 16,4,4096,32 -> 16,4,4096,64
            # 건쌤: y = [4,3] compare with e1(1,0),e2(0,1) ->dotproduct-> 4,3 -> output:0(e1)
            #           d=64, normalize안하고 ,y=[aX;-aX]
            #           why no normalize?: finding argmax does not matter
            #           why y=[aX;-aX] ?? -> e1,e2만보는게 아니라 -e1,-e2도고려하니 -aX!!
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1) # 이거에 argmax치면 rotation+cluster의미란겨 이리해주면..
            # 16,4,4096,64 -> 16, 16384(4*4096)
            # argmax(16,4,4096,64,dim=-1) -> [16,4,4096] - [0~63]중 하나나옴 마지막 칼럼이 이게 cluster class !
            #   argmax[1, 4, 0.5, 99] -> 3 나온다(99자리)
            buckets = torch.argmax(rotated_vecs, dim=-1)  # 이게 h(X) 값이다. [16,4,4096,1]
            # buckets is now (self.n_hashes, seqlen). Next we add offsets so that
            # bucket numbers from different hashing rounds don't overlap.
            offsets = torch.arange(self.n_hashes, device=device) # [0,1,2,3]
            offsets = torch.reshape(offsets * n_buckets, (1, -1, 1))
            # [0,1,2,3] -> [
            #               0,
            #               64,
            #               128,
            #               196]
            # offsets: 1D(4) -> 3D(1,4,1)
            # buckets: 16, 4, 4096 -> 각각보면, [4,4096], 4round니까 한줄[4096] 이게 개별인풋인거야
            #           여기다가 첫줄에는 0 더하고 1번쭐에는 64, ...., 이렇게 다 더해줘서 differentiate 를주는거야 그냥.. 각 라운드가 숫자가 달라지게.
            # a = buckets + offsets
            buckets = torch.reshape(buckets + offsets, (batch_size, -1,))
            #    bucket=16(멀티헤드배치사이즈),4(n해시라운드),4096 <- 각 4 라운드를 differentiate해줌. 숫자 더해줘서(0,64,128,192)
            #   뜻: 1st hash round:1-64, 2nd hash round:64-128, 3rd hash round:128-192 ...
        else:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            # In this configuration, we map each item to the top self.n_hashes buckets
            rotated_vecs = torch.squeeze(rotated_vecs, 0)
            bucket_range = torch.arange(rotated_vecs.shape[-1], device=device)
            bucket_range = torch.reshape(bucket_range, (1, -1))
            bucket_range = bucket_range.expand_as(rotated_vecs.shape)

            _, buckets = sort_key_val(rotated_vecs, bucket_range, dim=-1)
            buckets = buckets[:, -self.n_hashes:]

            h, *_ = buckets.shape
            buckets = torch.reshape(buckets.permute((*_, h)), (-1,))

        return buckets

    def forward(self, qk, v, query_len=None, input_mask=None, input_attn_mask=None, **kwargs):
        #                                 16,4096,64
        batch_size, seqlen, dim, device = *qk.shape, qk.device

        query_len = default(query_len, seqlen)  # 4096
        is_reverse = kwargs.pop('_reverse', False)
        depth = kwargs.pop('_depth', None)

        assert seqlen % (self.bucket_size * 2) == 0, \
            f'Sequence length ({seqlen}) needs to be divisible by target bucket size  x 2 - {self.bucket_size * 2}'

        n_buckets = seqlen // self.bucket_size  # n_bucket = 64..
        # 사실상은, [16,4,4096] 인데 계산 편의를 위해 [16,4*4096]==[16 16384]로해줌. 나중에 슬라이스하면됌.
        buckets = self.hash_vectors(n_buckets, qk, key_namespace=depth, fetch=is_reverse, set_cache=self.training)


        # We use the same vector as both a query and a key.
        assert int(buckets.shape[1]) == self.n_hashes * seqlen

        total_hashes = self.n_hashes

        # [16,0~4*4096] 2D 어레이 만들어짐.
        ticker = torch.arange(total_hashes * seqlen, device=device).unsqueeze(0).expand_as(buckets)
        aaa = ticker % seqlen  # [16,16384] -> 각 16줄을 보면 [0,1,2,..,4096, 0,..,4096, 0,..,4096, 0,..,4096]
        bbb = seqlen * buckets  # buckets=[16,16384] 에다가 각줄에 4096을 곱해줌

        # bucket([16,4*4096] 각 한줄한줄은 multihead(4)*batchsize(4) 의 각각의 인풋이야
        # ticker%4096 = [16, 0~4095가 4묶음] <- 이건 말됌. 4Round였으니까 이게 진짜 자리지...
        # each row(4096*4 size) of 16 rows 는 항상
        buckets_and_t = seqlen * buckets + (ticker % seqlen)
        buckets_and_t = buckets_and_t.detach()

        # Hash-based sort ("s" at the start of variable names means "sorted")
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = sort_key_val(sticker, ticker, dim=-1)
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
        bk = F.normalize(bqk, p=2, dim=-1).type(bq.type())

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

        # Mask for post qk attention logits of the input sequence
        if input_attn_mask is not None:
            input_attn_mask = F.pad(input_attn_mask,
                                    (0, seqlen - input_attn_mask.shape[-1], 0, seqlen - input_attn_mask.shape[-2]),
                                    value=True)
            dot_attn_indices = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            input_attn_mask = input_attn_mask.reshape(batch_size, -1)
            dot_attn_indices = dot_attn_indices.reshape(batch_size, -1)
            mask = input_attn_mask.gather(1, dot_attn_indices).reshape_as(dots)
            dots.masked_fill_(~mask, masked_value)
            del mask

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            input_mask = F.pad(input_mask, (0, seqlen - input_mask.shape[1]), value=True)
            mq = input_mask.gather(1, st).reshape((batch_size, chunk_size, -1))
            mkv = look_one_back(mq)
            mask = mq[:, :, :, None] * mkv[:, :, None, :]
            dots.masked_fill_(~mask, masked_value)
            del mask

        # Causal masking
        if self.causal:
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :]
            if seqlen > query_len:
                mask = mask & (bkv_t[:, :, None, :] < query_len)
            dots.masked_fill_(mask, masked_value)
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
        del self_mask

        # Mask out attention to other hash buckets.
        if not self._attend_across_buckets:
            bq_buckets = bkv_buckets = torch.reshape(sbuckets_and_t // seqlen, (batch_size, chunk_size, -1))
            bkv_buckets = look_one_back(bkv_buckets)
            bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
            dots.masked_fill_(bucket_mask, masked_value)
            del bucket_mask

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # There are two possible strategies here. (1) The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition. (2) When hard_k is set, the code
        # instead masks all but the first occurence of each query-key pair.
        if not self._allow_duplicate_attention:
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % chunk_size
            if not self._attend_across_buckets:
                locs1 = buckets * chunk_size + locs1
                locs2 = buckets * chunk_size + locs2
            locs = torch.cat([
                torch.reshape(locs1, (batch_size, total_hashes, seqlen)),
                torch.reshape(locs2, (batch_size, total_hashes, seqlen)),
            ], 1).permute((0, 2, 1))

            slocs = batched_index_select(locs, st)
            b_locs = torch.reshape(slocs, (batch_size, chunk_size, -1, 2 * total_hashes))

            b_locs1 = b_locs[:, :, :, None, :total_hashes]

            bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, total_hashes))
            bq_locs = torch.reshape(bq_locs, b_locs.shape)
            bkv_locs = look_one_back(b_locs)

            dup_counts = (bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :])
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = chunked_sum(dup_counts, chunks=(total_hashes * batch_size))
            dup_counts = dup_counts.detach()
            assert dup_counts.shape == dots.shape
            dots = dots - torch.log(dup_counts + 1e-9)
            del dup_counts

        # Softmax.
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp).type(dots.type())
        dropped_dots = self.dropout(dots)

        bo = torch.einsum('buij,buje->buie', dropped_dots, bv)
        so = torch.reshape(bo, (batch_size, -1, dim))
        slogits = torch.reshape(dots_logsumexp, (batch_size, -1,))

        class UnsortLogits(Function):
            @staticmethod
            def forward(ctx, so, slogits):
                so = so.detach()
                slogits = slogits.detach()
                o = batched_index_select(so, undo_sort)
                _, logits = sort_key_val(sticker, slogits, dim=-1)
                return o, logits

            @staticmethod
            def backward(ctx, grad_x, grad_y):
                so_grad = batched_index_select(grad_x, sticker)
                _, slogits_grad = sort_key_val(buckets_and_t, grad_y, dim=-1)
                return so_grad, slogits_grad

        o, logits = UnsortLogits.apply(so, slogits)
        o = torch.reshape(o, (batch_size, total_hashes, seqlen, dim))
        logits = torch.reshape(logits, (batch_size, total_hashes, seqlen, 1))

        if query_len != seqlen:
            query_slice = (slice(None), slice(None), slice(0, query_len))
            o, logits = o[query_slice], logits[query_slice]

        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        out = torch.sum(o * probs, dim=1)

        attn = torch.empty(0, device=device)

        # return unsorted attention weights
        if self._return_attn:
            attn_unsort = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            attn_unsort = attn_unsort.view(batch_size * total_hashes, -1).long()
            unsorted_dots = torch.zeros(batch_size * total_hashes, seqlen * seqlen, device=device)
            unsorted_dots.scatter_add_(1, attn_unsort, dots.view_as(attn_unsort))
            del attn_unsort
            unsorted_dots = unsorted_dots.reshape(batch_size, total_hashes, seqlen, seqlen)
            attn = torch.sum(unsorted_dots[:, :, 0:query_len, :] * probs, dim=1)

        # return output, attention matrix, and bucket distribution
        return out, attn, buckets


# local attention

class LocalAttention(nn.Module):
    def __init__(self, bucket_size, causal=False, look_backward=1, look_forward=0, dropout=0., shared_qk=False):
        super().__init__()
        assert not (causal and look_forward > 0), 'you cannot look forward if causal'
        self.bucket_size = bucket_size
        self.causal = causal
        self.look_backward = look_backward
        self.look_forward = look_forward
        self.shared_qk = shared_qk
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, input_mask=None):
        b, t, e, device, dtype = *q.shape, q.device, q.dtype
        bucket_size, causal, look_backward, look_forward, shared_qk = self.bucket_size, self.causal, self.look_backward, self.look_forward, self.shared_qk

        buckets = t // bucket_size

        if shared_qk:
            k = F.normalize(k, 2, dim=-1).type(q.type())

        ticker = torch.arange(t, device=device, dtype=dtype)[None, :]
        b_t = ticker.reshape(1, buckets, bucket_size)

        bucket_fn = lambda t: t.reshape(b, buckets, bucket_size, -1)
        bq, bk, bv = map(bucket_fn, (q, k, v))

        look_around_kwargs = {'backward': look_backward, 'forward': look_forward}
        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (e ** -0.5)
        mask_value = max_neg_value(dots)

        if shared_qk:
            mask = bq_t[:, :, :, None] == bq_k[:, :, None, :]
            dots.masked_fill_(mask, TOKEN_SELF_ATTN_VALUE)
            del mask

        if causal:
            mask = bq_t[:, :, :, None] < bq_k[:, :, None, :]
            dots.masked_fill_(mask, mask_value)
            del mask

        mask = bq_k[:, :, None, :] == -1
        dots.masked_fill_(mask, mask_value)
        del mask

        if input_mask is not None:
            h = b // input_mask.shape[0]
            input_mask = input_mask.reshape(-1, buckets, bucket_size)
            mq = mk = input_mask
            mk = look_around(mk, pad_value=False, **look_around_kwargs)
            mask = (mq[:, None, :, :, None] * mk[:, None, :, None, :])
            mask = merge_dims(0, 1, mask.expand(-1, h, -1, -1, -1))
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhje->bhie', attn, bv)
        out = out.reshape(b, t, e)
        return out


# simple full attention

class FullQKAttention(nn.Module):
    def __init__(self, causal=False, dropout=0.):
        super().__init__()
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

    def forward(self, qk, v, query_len=None, input_mask=None, input_attn_mask=None, **kwargs):
        b, seq_len, dim = qk.shape
        query_len = default(query_len, seq_len)
        t = query_len

        q = qk[:, 0:query_len]
        qk = F.normalize(qk, 2, dim=-1).type(q.type())

        dot = torch.einsum('bie,bje->bij', q, qk) * (dim ** -0.5)

        # qk attention requires tokens not attend to self
        i = torch.arange(t)
        dot[:, i, i] = TOKEN_SELF_ATTN_VALUE
        masked_value = max_neg_value(dot)

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            mask = input_mask[:, 0:query_len, None] * input_mask[:, None, :]
            mask = F.pad(mask, (0, seq_len - mask.shape[-1]), value=True)
            dot.masked_fill_(~mask, masked_value)

        # Mask for post qk attention logits of the input sequence
        if input_attn_mask is not None:
            input_attn_mask = F.pad(input_attn_mask, (0, seq_len - input_attn_mask.shape[-1]), value=True)
            dot.masked_fill_(~input_attn_mask, masked_value)

        # 페이퍼보면 max_neg_value를 lower triangle에다가 걸어버린다
        if self.causal:
            i, j = torch.triu_indices(t, t, 1)  # triangle upper파트만 계산
            dot[:, i, j] = masked_value

        dot = dot.softmax(dim=-1)
        dot = self.dropout(dot)

        out = torch.einsum('bij,bje->bie', dot, v)

        return out, dot, torch.empty(0)


# Shared qk attention, using either full or LSH attention

class LSHSelfAttention(nn.Module):
    def __init__(self, dim, heads=8,
                 bucket_size=64, n_hashes=8,
                 causal=False, attn_chunks=1,
                 random_rotations_per_head=False,
                 attend_across_buckets=True, allow_duplicate_attention=True,
                 num_mem_kv=0, one_value_head=False, use_full_attn=False,
                 full_attn_thres=None, return_attn=False, post_attn_dropout=0.,
                 dropout=0., n_local_attn_heads=0,
                 **kwargs):
        super().__init__()
        assert dim % heads == 0, 'dimensions must be divisible by number of heads'
        assert n_local_attn_heads < heads, 'local attention heads must be less than number of heads'

        self.dim = dim  # 512
        self.heads = heads  # 8
        self.attn_chunks = default(attn_chunks, 1)  # 디폴 1 뱉어라 Non이면

        self.v_head_repeats = (heads if one_value_head else 1) # facebooks persistent paper
        v_dim = dim // self.v_head_repeats

        self.toqk = nn.Linear(dim, dim, bias=False)  # shared q k
        self.tov = nn.Linear(dim, v_dim, bias=False)  # 512 -> 512
        self.to_out = nn.Linear(dim, dim)

        self.bucket_size = bucket_size
        # 이거씀..
        self.lsh_attn = LSHAttention(bucket_size=bucket_size, n_hashes=n_hashes, causal=causal,
                                     random_rotations_per_head=random_rotations_per_head,
                                     attend_across_buckets=attend_across_buckets,
                                     allow_duplicate_attention=allow_duplicate_attention,
                                     return_attn=return_attn, dropout=dropout, **kwargs)
        self.full_attn = FullQKAttention(causal=causal, dropout=dropout)  # 안씀 ...
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)

        self.use_full_attn = use_full_attn
        self.full_attn_thres = default(full_attn_thres, bucket_size)

        self.num_mem_kv = num_mem_kv
        self.mem_kv = nn.Parameter(torch.randn(1, num_mem_kv, dim, requires_grad=True)) if num_mem_kv > 0 else None

        self.n_local_attn_heads = n_local_attn_heads
        self.local_attn = LocalAttention(bucket_size=bucket_size * 2, causal=causal, dropout=dropout, shared_qk=True,
                                         look_forward=(1 if not causal else 0))

        self.callback = None

    def forward(self, x, keys=None, input_mask=None, input_attn_mask=None, context_mask=None, **kwargs):
        device, dtype = x.device, x.dtype
        # *는 리스트로 3개가 들어오나 ?? ㅇㅇ 맞네 4, 4096, 512
        b, t, e, h, m, l_h = *x.shape, self.heads, self.num_mem_kv, self.n_local_attn_heads

        # 4 0 512 아무변화가 없는데 ? mem_kv -> mem
        mem_kv = default(self.mem_kv, torch.empty(b, 0, e, dtype=dtype, device=device))
        mem = mem_kv.expand(b, m, e)

        keys = default(keys, torch.empty(b, 0, e, dtype=dtype, device=device))
        # keys = 4,0,512
        c = keys.shape[1]

        kv_len = t + m + c
        use_full_attn = self.use_full_attn or kv_len <= self.full_attn_thres  # 이건무엇??

        x = torch.cat((x, mem, keys), dim=1)  # 4, 4096, 512 우리 우선 mem 은 안씀 ㅎ
        qk = self.toqk(x)  # 4 4096 512
        v = self.tov(x)  # 4 4096 512
        v = v.repeat(1, 1, self.v_head_repeats)  # 우선 우리 의미없다 이건.. repeat = unlike expand, deep copy the tensor's data

        def merge_heads(v):
            # 4  4096   8
            return v.view(b, kv_len, h, -1).transpose(1, 2)

        def split_heads(v):
            return v.view(b, h, t, -1).transpose(1, 2).contiguous()

        merge_batch_and_heads = partial(merge_dims, 0, 1)

        qk, v = map(merge_heads, (qk, v))  # qk=4,8,4096,64,   v=4 8 4096 64

        has_local = l_h > 0
        lsh_h = h - l_h

        split_index_fn = partial(split_at_index, 1, l_h)
        (lqk, qk), (lv, v) = map(split_index_fn, (qk, v))
        lqk, qk, lv, v = map(merge_batch_and_heads, (lqk, qk, lv, v))

        masks = {}
        if input_mask is not None or context_mask is not None:
            default_mask = torch.tensor([True], device=device)
            i_mask = default(input_mask, default_mask.expand(b, t))
            m_mask = default_mask.expand(b, m)
            c_mask = default(context_mask, default_mask.expand(b, c))
            mask = torch.cat((i_mask, m_mask, c_mask), dim=1)
            mask = merge_batch_and_heads(expand_dim(1, lsh_h, mask))
            masks['input_mask'] = mask

        if input_attn_mask is not None:
            input_attn_mask = merge_batch_and_heads(expand_dim(1, lsh_h, input_attn_mask))
            masks['input_attn_mask'] = input_attn_mask

        attn_fn = self.lsh_attn if not use_full_attn else self.full_attn
        partial_attn_fn = partial(attn_fn, query_len=t, **kwargs)
        attn_fn_in_chunks = process_inputs_chunk(partial_attn_fn, chunks=self.attn_chunks)

        out, attn, buckets = attn_fn_in_chunks(qk, v, **masks)

        if self.callback is not None:
            self.callback(attn.reshape(b, h, t, -1), buckets.reshape(b, h, -1))

        if has_local:
            lq = lk = lqk[:, :t]
            local_out = self.local_attn(lq, lk, lv, input_mask=input_mask)
            local_out = local_out.reshape(b, l_h, t, -1)
            out = out.reshape(b, lsh_h, t, -1)
            out = torch.cat((local_out, out), dim=1)

        out = split_heads(out).view(b, t, e)
        out = self.to_out(out)
        return self.post_attn_dropout(out)


# feed forward

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., activation=None, glu=False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x


# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)  # 4096 -> 512
        self.emb.weight.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        # t를 인풋사이즈(4096)니까 0~ 4095 제너레이트 !
        t = torch.arange(x.shape[1], device=x.device)  # python의 레인지랑같음
        aaa = self.emb(t)
        return self.emb(t)  # 4096 x 512 사이즈 # 4096포지션 하나하나 512 사이즈 벡터준거야


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device).type(self.inv_freq.type())
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]


# rewritten from https://github.com/google/trax/blob/master/trax/layers/attention.py#L126
# with help from @AranKomat
class AxialPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len, axial_shape=(), axial_emb_dims=()):
        super().__init__()
        assert sum(axial_emb_dims) == dim, 'axial position embedding dimensions must sum to model dimension'
        assert reduce(mul, axial_shape,
                      1) == max_seq_len, 'axial position shape must multiply up to max sequence length'

        self.seq_len = max_seq_len
        self.shape = axial_shape
        self.emb_dims = axial_emb_dims

        self.weights = nn.ParameterList([])
        for ind, (d_emb, shape) in enumerate(zip(self.emb_dims, self.shape)):
            ax_shape = [1] * len(self.shape)
            ax_shape[ind] = shape
            ax_shape = (1, *ax_shape, d_emb)
            ax_emb = nn.Parameter(torch.zeros(ax_shape).normal_(0, 1))
            self.weights.append(ax_emb)

    def forward(self, x):
        b, t, e = x.shape
        embs = []

        for ax_emb in self.weights:
            ax_emb_dim = ax_emb.shape[-1]
            expand_shape = (b, *self.shape, ax_emb_dim)
            emb = ax_emb.expand(expand_shape).reshape(b, self.seq_len, ax_emb_dim)
            embs.append(emb)

        pos_emb = torch.cat(embs, dim=-1)
        return pos_emb[:, :t]


# reformer lm

class Reformer(nn.Module):
    def __init__(self, dim, depth, max_seq_len, heads=8,
                 bucket_size=64, n_hashes=8, ff_chunks=100,
                 attn_chunks=None, causal=False, weight_tie=False,
                 lsh_dropout=0., ff_dropout=0., ff_activation=None,
                 ff_mult=4, ff_glu=False, post_attn_dropout=0., layer_dropout=0.,
                 lsh_attend_across_buckets=True, lsh_allow_duplicate_attention=True,
                 random_rotations_per_head=False, twin_attention=False, use_scale_norm=False,
                 use_rezero=False, use_full_attn=False, full_attn_thres=0, reverse_thres=0,
                 num_mem_kv=0, one_value_head=False, n_local_attn_heads=0):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.bucket_size = bucket_size
        self.num_mem_kv = num_mem_kv

        self.twin_attention = twin_attention
        self.full_attn_thres = full_attn_thres

        get_attn = lambda: LSHSelfAttention(dim, heads, bucket_size, n_hashes,
                                            causal=causal, dropout=lsh_dropout,
                                            post_attn_dropout=post_attn_dropout,
                                            attn_chunks=attn_chunks,
                                            allow_duplicate_attention=lsh_allow_duplicate_attention,
                                            attend_across_buckets=lsh_attend_across_buckets,
                                            random_rotations_per_head=random_rotations_per_head,
                                            num_mem_kv=num_mem_kv, use_full_attn=use_full_attn,
                                            full_attn_thres=full_attn_thres, one_value_head=one_value_head,
                                            n_local_attn_heads=n_local_attn_heads)
        get_ff = lambda: FeedForward(dim, dropout=ff_dropout, activation=ff_activation, mult=ff_mult, glu=ff_glu)

        if weight_tie:
            get_attn = cache_fn(get_attn)
            get_ff = cache_fn(get_ff)

        blocks = []

        # 배치놈은 4*4096 를 민구하는거야 - 비젼에서 Channel = nlp에서의 512벡터사이즈(model_dim)
        norm_type = ScaleNorm if use_scale_norm else nn.LayerNorm  # 4, 4096,512 -> mean 은 4096,512 개의 민 구함
        # norm_type = ScaleNormMatt if use_scale_norm else nn.LayerNorm  # 4, 4096,512 -> mean 은 4096,512 개의 민 구함

        residual_fn_wrapper = ReZero if use_rezero else partial(PreNorm, norm_type, dim)

        for _ in range(depth):
            attn = get_attn()  # 캐시되있는 LSHSelfAttention() 가 계속 나옴. 걍 같은 파라미터 계속씀 웨이트 타이면!
            parallel_net = get_attn() if twin_attention else get_ff()

            # revnet에서 두줄기를 이어주는 그 교차되는곳 !
            f = residual_fn_wrapper(attn)  # self attn 의 아웃풋나오면(attn) 그거로 scale Norm !
            g = residual_fn_wrapper(parallel_net)  # FF

            if not twin_attention and ff_chunks > 1:
                g = ChunkBeforeFF(ff_chunks, g, along_dim=-2) # self attention 끝나고 FeedForward구하는부분!

            blocks.append(nn.ModuleList([f, g]))  # f -> g 로 연결!

        # 여기가 forward펑션 커스터마이즈해줌
        # 이라인 전까지는 f -> g 이거밖에모름. 레브넷 투플로우 몰라
        self.layers = ReversibleSequence(nn.ModuleList(blocks), layer_dropout=layer_dropout,
                                         reverse_thres=reverse_thres, send_signal=True)

    def forward(self, x, **kwargs):
        # x=4,4096,512 -> 4,4096,512*2 레브넷 위해 하나 복사해논거임
        # x = 4,4096,512
        # x'= 4,4096,512 <-이거 두개 컨캣 !
        # 즉 (4*4096*512)*2로 쪼갤수있다
        x = torch.cat([x, x], dim=-1)
        arg_route = (True, self.twin_attention)
        x = self.layers(x, arg_route=arg_route, **kwargs)
        return torch.stack(x.chunk(2, dim=-1)).mean(dim=0)


class ReformerLM(nn.Module):
    def __init__(self, num_tokens, dim, depth, max_seq_len, heads=8, bucket_size=64, n_hashes=4, ff_chunks=100,
                 attn_chunks=1, causal=False, weight_tie=False, lsh_dropout=0., ff_dropout=0., ff_mult=4,
                 ff_activation=None, ff_glu=False, post_attn_dropout=0., layer_dropout=0.,
                 random_rotations_per_head=False, twin_attention=False, use_scale_norm=False, use_rezero=False,
                 use_full_attn=False, full_attn_thres=0, reverse_thres=0, num_mem_kv=0, one_value_head=False,
                 emb_dim=None, return_embeddings=False, weight_tie_embedding=False, fixed_position_emb=False,
                 axial_position_emb=False, axial_position_shape=(), axial_position_dims=(), n_local_attn_heads=0):
        super().__init__()
        emb_dim = default(emb_dim, dim)  # emb_dim=512, dim=512
        self.max_seq_len = max_seq_len

        # look up table
        # 256 * 512
        self.token_emb = nn.Embedding(num_tokens, emb_dim)

        # 임베딩매트릭스 초기값 이니셜라이제이션을 -0.01 ~ 0.01 안에서 쓸꺼다 (트레인할꺼임)
        # 가우시안 /mu 0, sigma 0.01 -> entorpy가 1.35니까
        # 유니폼으로 이니셜라이즈하면 그거보단작을거야
        # 즉 애초에 시작할떄 엔트로피를좀 작게시작하자이거.. 국룰.....
        self.token_emb.weight.data.uniform_(-0.01, 0.01)  # float64

        # Identyty() = nn.Linear(512,512)
        # emb_dim=인풋 벡터딤, dim=아웃풋 벡터딤
        self.to_model_dim = Identity() if emb_dim == dim else nn.Linear(emb_dim, dim)

        if axial_position_emb:
            self.pos_emb = (emb_dim, max_seq_len, axial_position_shape, axial_position_dims)
        elif fixed_position_emb:
            self.pos_emb = FixedPositionalEmbedding(emb_dim)
        else:  # 우리가 코드현재상 이거 (노말한 포지셔널임베딩씀) 근데 axial_position(리뷰중인논문) 이게 더 좋다카더라
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len)

        self.reformer = Reformer(dim, depth, max_seq_len, heads=heads, bucket_size=bucket_size, n_hashes=n_hashes,
                                 ff_chunks=ff_chunks, attn_chunks=attn_chunks, causal=causal, weight_tie=weight_tie,
                                 lsh_dropout=lsh_dropout, ff_mult=ff_mult, ff_activation=ff_activation, ff_glu=ff_glu,
                                 ff_dropout=ff_dropout, post_attn_dropout=0., layer_dropout=layer_dropout,
                                 random_rotations_per_head=random_rotations_per_head, twin_attention=twin_attention,
                                 use_scale_norm=use_scale_norm, use_rezero=use_rezero, use_full_attn=use_full_attn,
                                 full_attn_thres=full_attn_thres, reverse_thres=reverse_thres, num_mem_kv=num_mem_kv,
                                 one_value_head=one_value_head, n_local_attn_heads=n_local_attn_heads)

        if return_embeddings:
            self.out = Identity()
            return

        self.out = nn.Sequential(
            nn.Linear(dim, emb_dim) if emb_dim != dim else Identity(),  # nn.Linear(512,512) 들어온거나 마찬가지 현재상황은
            # weight_tie_embedding: 일종의 레귤러라이제이션 역활
            # token_emb(256 * 512)                                                              #256 512
            nn.Linear(emb_dim, num_tokens) if not weight_tie_embedding else MatrixMultiply(self.token_emb.weight,
                                                                                           transpose=True,
                                                                                           normalize=True)
        )

    def forward(self, x, **kwargs):
        x = self.token_emb(x)
        x = x + self.pos_emb(x).type(x.type())  # x=float, pos_emb=torch float64 // 4096,512
        pos_emb = self.pos_emb(x)
        x = self.to_model_dim(x)  # 4,4096,512
        x = self.reformer(x, **kwargs)

        # 4,4096,512 -> 4* 4096 * 256 (액티베이션 결과값!) 이게이제 나중에 cross_entropy들어가서 소프트맥스 되고 골라질거다
        # weight_tie_embedding 를 안쓰면 그냥 nn.Linear로 모델을 더 복잡하게해서 구해줄수있다
        #   단점: parameter up -> 모델이 이해할수 있는 모델의 컴플렉시티/엔트로피 업 (ex: y=ax+b vs y=ax^100+..+b 면 후자가 더 복잡)
        #                               ** 엔트로피=컴플렉시티 메져
        #                           -> over fitting
        #  그런데 weight_tie_embedding 파라미터가 증거하지않는다!
        #       어케? 내가 각 이폭마다 트레인해줬던 look up table 을 디코더마냥 가져다가 쓸거다 #논문: Using the output Embedding to Improve Language Models
        aaa = self.out(x);
        return self.out(x)
