import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states

# following example for saving and setting rng here https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    # rng = random number generator !
    def record_rng(self, *args):
        # 모든 포워드가 돌때, 랜덤시드를 저장해놈 to reproduce the random initialization as same as the recorded
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng = False, set_rng = False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)

# heavily inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
# once multi-GPU is confirmed working, refactor and send PR back to source
class ReversibleBlock(nn.Module):
    def __init__(self, f, g, depth=None, send_signal = False):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

        self.depth = depth
        self.send_signal = send_signal

    def forward(self, x, f_args = {}, g_args = {}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1, y2 = None, None

        if self.send_signal:  # 실험 로깅용
            f_args['_reverse'] = g_args['_reverse'] = False
            f_args['_depth'] = g_args['_depth'] = self.depth

        with torch.no_grad():  # 레브넷 핵심 ! 그라디언트 킵 안할꺼야 !!!!!!!!
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)

        return torch.cat([y1, y2], dim=2)

    # y = torch.cat([y1, y2], dim=2)
    def backward_pass(self, y, dy, f_args = {}, g_args = {}):
        y1, y2 = torch.chunk(y, 2, dim=2)  # 쪼개요~
        del y  # 가져온거지움

        dy1, dy2 = torch.chunk(dy, 2, dim=2)  # dy = 이전스텝에서 리턴된 데리베티브 of Y
        del dy

        if self.send_signal:
            f_args['_reverse'] = g_args['_reverse'] = True
            f_args['_depth'] = g_args['_depth'] = self.depth

        with torch.enable_grad():  # 뱍워드할때만 그라디언트 계산함
            y1.requires_grad = True  # 논문의 z1임 ㅎㅎ, y1이 그라디언트를 갖을수있다 / 백프랍할수있다(변화가있다)
            gy1 = self.g(y1, set_rng=True, **g_args)  # step (3)'s G(z_1)
            torch.autograd.backward(gy1, dy2)  # step (9)
            #                     dG/dWg*y2바, dG/dy1 * y2바 를 구함, 후자는 y1.grad에 박힘, 전자는 Wg애들한테 박힘
            #             dG wrt (self.g의 모든 웨이트들(Wg) & activation(y1))gy1구할때 쓰여지는 모든 웨이트들 을구하고 dy2랑 곱함
            #
        with torch.no_grad():
            x2 = y2 - gy1  # step(3)
            del y2, gy1

            dx1 = dy1 + y1.grad  #  5,7합침.
            del dy1
            y1.grad = None # 이제 안쓸꺼니까 지워주

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)
            # x2.requires_grad 해서 인에이블 그라드해주고
            # backward해주면 x2웨이트 업데이트에할놈 구해져서 x2.grad에 저장해줌
            # dF/dWf*dx1, dF/dx2*dx1: 후자가 x2.grad, 전자가 Wf업데이트에 쓰일놈들

        with torch.no_grad():
            x1 = y1 - fx2  # step(4)
            del y1, fx2

            dx2 = dy2 + x2.grad  # step(6)
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=2) # x2.requires_grad 를 꺼줌.
            dx = torch.cat([dx1, dx2], dim=2)

        return x, dx

class IrreversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, x, f_args, g_args):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1 = x1 + self.f(x2, **f_args)
        y2 = x2 + self.g(y1, **g_args)
        return torch.cat([y1, y2], dim=2)

class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y = x.detach() # 여기서 결과값을 ctx.y에 저장해놓고 backward에서 y로가져다쓴다.
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y # 백워드 시작할때 최종결과값 오브 포워드 가져오기
        kwargs = ctx.kwargs # 디버깅용도 ㅎㅎ
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None


class ReversibleSequence(nn.Module):
    def __init__(self, blocks, layer_dropout = 0., reverse_thres = 0, send_signal = False):
        super().__init__()
        self.layer_dropout = layer_dropout
        self.reverse_thres = reverse_thres

        self.blocks = nn.ModuleList(
                # 여기서 forward, backward를 레브넷용으로 짜놈..
                [ReversibleBlock(f, g, depth, send_signal) for depth, (f, g) in enumerate(blocks)]
            )
        self.irrev_blocks = nn.ModuleList([IrreversibleBlock(f=f, g=g) for f, g in blocks])

    def forward(self, x, arg_route = (True, True), **kwargs):
        reverse = x.shape[1] > self.reverse_thres
        blocks = self.blocks if reverse else self.irrev_blocks

        if self.training and self.layer_dropout > 0:
            to_drop = torch.empty(len(self.blocks)).uniform_(0, 1) < self.layer_dropout
            blocks = [block for block, drop in zip(self.blocks, to_drop) if not drop]
            blocks = self.blocks[:1] if len(blocks) == 0 else blocks

        f_args, g_args = map(lambda route: kwargs if route else {}, arg_route)
        block_kwargs = {'f_args': f_args, 'g_args': g_args}

        if not reverse:
            for block in blocks:
                x = block(x, **block_kwargs)
            return x

        # 이게 forward, backward를 레브넷 용으로 짜논걸 오버라이드해준거임(ReversibleBlcok에서 짠거)
        # 잘은 모르지만... 이 apply를 해줘야 forward, backward가 ReversibleBlock에서 짜논걸로 오버라이드 된다 ...
        return _ReversibleFunction.apply(x, blocks, block_kwargs)


    '''
    self.blocks = nn.ModuleList(
                # 여기서 forward, backward를 레브넷용으로 짜놈..
                [ReversibleBlock(f, g, depth, send_signal) for depth, (f, g) in enumerate(blocks)]
            )
    f,g를 이렇게 내가 짜논 forward, backward있는 ReversibleBlock클래스로 감사쭈고 그걸 모듈리스트해준담에
    
    def forward:
    여기에서 
    _ReversibleFunction.apply(x, blocks, block_kwargs)
    이걸 해줘야 비로소 내가원하는 레브넷 백워드 포워드 어플라이가 된다 ! 
    
    '''
