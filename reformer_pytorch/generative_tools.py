from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from reformer_pytorch.reformer_pytorch import ReformerLM
from reformer_pytorch.autopadder import Autopadder

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.gather(1, sorted_indices)

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

class TrainingWrapper(nn.Module): # 토치 클래스 상속! 즉 포워드를 봐야겠지 ?
    def __init__(self, net, ignore_index = -100, pad_value = 0):
        super().__init__()
        assert isinstance(net, ReformerLM), 'generative trainer wrapper can only accept ReformerLM class'
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = Autopadder(net)
        self.max_seq_len = net.max_seq_len

    # 벨리데이션 할때 쓸꺼야 - 그라디언트가 필요없다니 자연스러운 추측가능!
    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token = None, temperature = 1., filter_logits_fn = top_k, filter_thres = 0.9, **kwargs):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        input_mask = kwargs.pop('input_mask', None)

        if input_mask is None:
            input_mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            input_mask = input_mask[:, -self.max_seq_len:]

            logits = self.net(x, input_mask=input_mask, **kwargs)[:, -1, :]
            filtered_logits = filter_logits_fn(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            input_mask = F.pad(input_mask, (0, 1), value=True)

            if eos_token is not None and (sample == eos_token).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def forward(self, x, return_loss = False, **kwargs):
        # partial <- 미리 변수값 몇개를 너주는거야 !
        # pad_sequence 펑션에 파라미트 batch_first, padding_value 를 디폴밸류 정해준거
        pad = partial(pad_sequence, batch_first = True, padding_value = self.pad_value)

        if not return_loss: # 로쓰 필요없을때(벨리데이션할때) !
            if not isinstance(x, torch.Tensor):
                x = pad(x)
            return self.net(x, **kwargs)

        if isinstance(x, torch.Tensor):
            xi = x[:, :-1]
            xo = x[:, 1:]
        else:
            xi = pad(list(map(lambda t: t[:-1], x)))
            xo = pad(list(map(lambda t: t[1:], x)))

        # size = [4, 4096, 256] <- 256 개 다음 캐릭터 후보들!
        out = self.net(xi, **kwargs) # net = 잘싸논 리포머 전체 !! 인코더 디코더 합쳐져있음 enwiki8에선

        # out.transpose 하면 사이즈를 [4, 256, 4096]
        # cross_entropy 가 그냥 [batch_size, softmax_size, 나머지...] 요구함
        loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index = self.ignore_index)
        return loss
