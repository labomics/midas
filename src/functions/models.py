import numpy as np
import torch as th
from torch.autograd import Function



class CheckBP(Function):
    @staticmethod
    def forward(ctx, input, label, show):
        ctx.label = label
        ctx.show = show
        # print(ctx.label + ': forward passed.')
        # print(input)
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        grad_mean = grad_output.abs().mean().item()
        if ctx.show == 1:
            print('grad_' + ctx.label + ': ' + str(grad_mean))

        # assert grad_mean < 10, 'Abnormal gradients for ' + ctx.label + ': ' + str(grad_mean)
        # if grad_mean > 10:
        #     print('grad_' + ctx.label + ' is LARGE: ' + str(grad_mean))
        # if grad_mean != grad_mean:
        #     print('grad_' + ctx.label + ' is NAN!')
        return grad_output, None, None


class Identity(Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Round(Function):
    @staticmethod
    def forward(ctx, input):
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class StraightThrough(Function):
    @staticmethod
    def forward(ctx, input):
        u = th.rand(input.size()).cuda()
        return u.lt(input).type_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ArgMax(Function):
    @staticmethod
    def forward(ctx, input):
        _, max_index = input.max(1) # N
        output = input.clone().zero_() # N * K
        return output.scatter_(1, max_index.unsqueeze(1), 1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output