import math
import torch
from torch import nn
import torch.nn.functional as F

from reformer_pytorch.reformer_pytorch import Reformer, ReformerLM, LSHSelfAttention

def pad_to_multiple(tensor, seqlen, multiple, dim=-1):
    m = seqlen / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=0)

class Autopadder(nn.Module):
    def __init__(self, net):
        super().__init__()
        # 이중에 하나!!
        assert isinstance(net, (LSHSelfAttention, Reformer, ReformerLM)), 'only modules LSHSelfAttention, Reformer, ReformerLM accepted'
        self.net = net

        reformer = net.reformer if isinstance(net, ReformerLM) else net
        self.pad_dim = -1 if isinstance(net, ReformerLM) else -2

        self.bucket_size = reformer.bucket_size
        self.num_mem_kv = reformer.num_mem_kv
        self.full_attn_thres = reformer.full_attn_thres

    def forward(self, x, **kwargs): # 딕셔너리형태 크와그
        b, t, m, device = *x.shape[:2], self.num_mem_kv, x.device

        keys = kwargs.get('keys')

        # 인풋토큰에다가 마스킹하는것(마스크도 하나의 단어다 !)
        # 마스크라는 의미로 들어가서 마스크 VS 단어들 해서 어텐션 구함.
        input_mask = kwargs.get('input_mask') # 단어 몇개에 마스크 해줌

        # 만약에 버트로 예를들면 어텐션을 Q,K,V 구할때
        # [I love MASK] <- MASK단어를 트레이닝할때 안쓰고, 맞추겠다
        # 즉 아웃풋이없고 내가 마스킹으로 아웃풋을 만들때 쓰여질 놈이야 !
        # 인코더,디코더에서 리포머 페이퍼보면, 어텐션이 모두에게들어가지않아. 어텐션 걸릴수 있는 방향에 리밋주는거임!
        #   예를들면 이제 뒤에놈한테는 안걸리게 !
        input_attn_mask = kwargs.get('input_attn_mask')

        k_len = 0 if keys is None else keys.shape[1]
        seqlen = t + m + k_len

        if seqlen > self.full_attn_thres:

            # enwiki8 에서는 input_mask==None 이고
            # 어텐션 쭉 다 계산한다
            # enwik8 :: input_mask = [True, True ... , True] *all true
            #        :: input_attn_mask = None
            if input_mask is None:
                input_mask = torch.full_like(x, True, device=x.device, dtype=torch.bool)

            x = pad_to_multiple(x, seqlen, self.bucket_size * 2, dim=self.pad_dim)

            if input_mask is not None:
                new_mask = F.pad(input_mask, (0, x.shape[1] - input_mask.shape[1]), value=False)
                kwargs.update(input_mask=new_mask)

            if input_attn_mask is not None:
                offset = x.shape[1] - input_attn_mask.shape[1]
                new_mask = F.pad(input_attn_mask, (0, offset, 0, offset), value=False)
                kwargs.update(input_attn_mask=new_mask)

        out = self.net(x, **kwargs)
        return out[:, 0:t]
