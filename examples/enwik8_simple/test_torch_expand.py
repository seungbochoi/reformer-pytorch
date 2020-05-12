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

def matrix_mult(a, b):
    print(a*b)

def padding(a):
    input_mask = a
    print(F.pad(a, (2,3),value=True))

def broad_cast(a,b):
    # print(a.shape)
    # print(b.shape)
    print(a)
    print(b)
    #
    print(a<b)
    mask = a<b
    dots = torch.ones(a.shape[0], b.shape[1])
    dots.masked_fill_(mask, -1000)
    print(dots)


if __name__ == "__main__":
    # x = [[1,2,3],[4,5,6]] -> x.expand(3,-1,-1,-1) ->
    # x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # expand_test(x)

    a = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    b = torch.tensor([3, 4, 5])
    c = torch.tensor([1, 5, 1, 9, 9, 9])
    # matrix_mult(a, b)
    # padding(a)
    broad_cast(b[:, None], c[None, :])