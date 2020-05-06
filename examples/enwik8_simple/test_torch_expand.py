import torch
import torch.nn as nn
import torch.nn.functional as F

def expand_test(x):
    xx = x.expand(3, -1, -1)
    print(xx)
    print(xx.shape)
    xx[0][0][0] = 10
    print(xx)
    print(x)

    input = torch.randn(5, 10)
    a = nn.Linear(10, 5)
    b = nn.Parameter(torch.randn(10, 5))  # learnable parameter matrix size 10 x 5
    sss = input.matmul(b)
    ssss = F.linear(input, b.transpose(1, 0))
    ssss_type = type(ssss)
    assert sss.detach().numpy() == ssss
    aaa = a(input)


if __name__ == "__main__":
    # x = [[1,2,3],[4,5,6]] -> x.expand(3,-1,-1,-1) ->
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    expand_test(x)
