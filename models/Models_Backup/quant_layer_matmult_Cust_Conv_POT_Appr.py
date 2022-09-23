import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
#########################
####### Custom convolution with mat mult and 2^w-2^-w approximation (Even backward pass) 93 % accuracy obtained
##########################
# PoT terms
def build_POT_value(B=2, additive=True):
    base_a = [0.]
    for i in range(B):
        base_a.append(2 ** (-i))
    values = []
    for a in base_a:
        values.append((a))
    values = torch.Tensor(list(set(values)))
    #values = values.mul(1.0 / torch.max(values))
    return values

# this function construct an additive pot quantization levels set, with clipping threshold = 1,
def build_power_value(B=2, additive=True):
    base_a = [0.]
    base_b = [0.]
    base_c = [0.]
    if additive:
        if B == 2:
            for i in range(3):
                base_a.append(2 ** (-i - 1))
        elif B == 4:
            for i in range(3):
                base_a.append(2 ** (-2 * i - 1))
                base_b.append(2 ** (-2 * i - 2))
        elif B == 6:
            for i in range(3):
                base_a.append(2 ** (-3 * i - 1))
                base_b.append(2 ** (-3 * i - 2))
                base_c.append(2 ** (-3 * i - 3))
        elif B == 3:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-i - 1))
                else:
                    base_b.append(2 ** (-i - 1))
                    base_a.append(2 ** (-i - 2))
        elif B == 5:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-2 * i - 1))
                    base_b.append(2 ** (-2 * i - 2))
                else:
                    base_c.append(2 ** (-2 * i - 1))
                    base_a.append(2 ** (-2 * i - 2))
                    base_b.append(2 ** (-2 * i - 3))
        else:
            pass
    else:
        for i in range(2 ** B - 1):
            base_a.append(2 ** (-i - 1))
    values = []
    for a in base_a:
        for b in base_b:
            for c in base_c:
                values.append((a + b + c))
    values = torch.Tensor(list(set(values)))
    values = values.mul(1.0 / torch.max(values))
    return values


def weight_quantization(b, grids, power=True):

    def uniform_quant(x, b):
        xdiv = x.mul((2 ** b - 1))
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    def power_quant(x, value_s):
        shape = x.shape
        #print('shape of x: ', shape)
        xhard = x.view(-1)
        #print('shape of xhard: ', xhard.shape)
        value_s = value_s.type_as(x)
        #print('shape of value_s: ', value_s)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]  # project to nearest quantization level
        xhard = value_s[idxs].view(shape)
        # xout = (xhard - x).detach() + x
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input.div_(alpha)                          # weights are first divided by alpha
            input_c = input.clamp(min=-1, max=1)       # then clipped to [-1,1]
            sign = input_c.sign()
            input_abs = input_c.abs()
            if power:
                input_q = power_quant(input_abs, grids).mul(sign)  # project to Q^a(alpha, B)
            else:
                input_q = uniform_quant(input_abs, b).mul(sign)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)               # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()             # grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            i = (input.abs()>1.).float()
            sign = input.sign()
            grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()
            return grad_input, grad_alpha

    return _pq().apply


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit, power=True):
        super(weight_quantize_fn, self).__init__()
        assert (w_bit <=5 and w_bit > 0) or w_bit == 32
        self.w_bit = w_bit-1
        self.power = power if w_bit>2 else False
        self.grids = build_POT_value(self.w_bit, additive=True)
        self.weight_q = weight_quantization(b=self.w_bit, grids=self.grids, power=self.power)
        self.register_parameter('wgt_alpha', Parameter(torch.tensor(3.0)))

    def forward(self, weight):
        weight_q = weight
        #if self.w_bit == 32:
        #    weight_q = weight
        #else:
        #    mean = weight.data.mean()
         #   std = weight.data.std()
         #   weight = weight.add(-mean).div(std)      # weights normalization or zero center the weights
        #    weight_q = self.weight_q(weight, self.wgt_alpha)
        return weight_q


def act_quantization(b, grid, power=True):

    def uniform_quant(x, b=3):
        xdiv = x.mul(2 ** b - 1)
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    def power_quant(x, grid):
        shape = x.shape
        xhard = x.view(-1)
        value_s = grid.type_as(x)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
        xhard = value_s[idxs].view(shape)
        return xhard

    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input=input.div(alpha)
            input_c = input.clamp(max=1)
            if power:
                input_q = power_quant(input_c, grid)
            else:
                input_q = uniform_quant(input_c, b)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            i = (input > 1.).float()
            grad_alpha = (grad_output * (i + (input_q - input) * (1 - i))).sum()
            grad_input = grad_input*(1-i)
            return grad_input, grad_alpha

    return _uq().apply

#@title rounding function POT
def pure_PoT(w_in):
    forward_value = 2**(w_in) - 2**(-w_in)
    return forward_value 

        
class F_alpha(torch.autograd.Function):
    """Both forward and backward are static methods."""

    @staticmethod
    def forward(ctx, weights):
        """
        In the forward pass we receive a Tensor containing the input (alpha) and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(weights)
        ctx.weights = weights
        # Add the modifications to weight here after it is saved for backward pass
        
        return 2**(weights) - 2**(-weights) 
#        return alpha*weights

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the inputs: here input (alpha) and weights
        """
        weights = ctx.saved_tensors
        #a = torch.tensor(2)
        
        grad_weights = torch.log(torch.tensor(2.0)) * (torch.tensor(2.0)**(ctx.weights) + torch.tensor(2.0)**(-ctx.weights)) * grad_output
        return grad_weights


#@title rounding function
def round_func_BPDA(input):
    # This is equivalent to replacing round function (non-differentiable) with
    # an identity function (differentiable) only when backward.
    n_digits=0
    forward_value = (input * 10**n_digits).round() / (10**n_digits)
    #forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out
    
#@title rounding function fixed
def round_binay_fixed(input, No_bits = 4):
    # This is equivalent to replacing round function (non-differentiable) with
    # an identity function (differentiable) only when backward.
#     No_bits=0
    forward_value = (input * 2**No_bits).round() / (2**No_bits)
    # insert regression to approximate the input to pot
    #forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out

#@title rounding function POT //
def round_log_func_PoT(w_in):
    w_sign = torch.sign(w_in)
#     forward_value = w_sign*2**(torch.log2(torch.abs(w_in)))  # approximating the 
    # forward_value = torch.where(w_in == 0, w_in, w_sign*2**(torch.log2(torch.abs(w_in))))
    forward_value = torch.where(w_in == 0, w_in, w_sign*2**(round_func_BPDA(torch.log2(torch.abs(w_in)))))
#     forward_value = w_sign*2**(round_func_BPDA(torch.log2(torch.abs(w_in))))  # approximating the 
#     print(forward_value.shape)
    out = w_in.clone()
    out.data = forward_value.data
    return out 


class QuantConv2d(nn.Module):  #Custom function
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=0, bias=True):
        super(QuantConv2d, self).__init__()
        self.k = kernel_size
        self.in_c = in_channel
        self.out_c = out_channel
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.ones(out_channel, in_channel, kernel_size, kernel_size)) 
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        #nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.fn = F_alpha.apply
    def forward(self, inp):
        # print(inp.shape)
        k = self.k
        stride = self.stride
        h_in, w_in = inp.shape[2], inp.shape[3]

        padding = self.padding  # + k//2
        batch_size = inp.shape[0]

        h_out = (h_in + 2 * padding - (k - 1) - 1) / stride + 1
        w_out = (w_in + 2 * padding - (k - 1) - 1) / stride + 1
        h_out, w_out = int(h_out), int(w_out)

        inp_unf = torch.nn.functional.unfold(inp, (k, k), padding=padding, stride = self.stride)
        # out_unf = inp_unf.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2)
        # print(inp_unf.shape)
#         in_feature = inp_unf.transpose(1, 2)
        in_feature = round_binay_fixed(inp_unf.transpose(1, 2), No_bits = 8)
#         weight_feature = self.weight.view(self.weight.size(0), -1).t()
        
#         weight_feature = (2**(s_w) - 2**(-s_w))/s
        #weight_feature = round_log_func_PoT(self.weight.view(self.weight.size(0), -1).t())
        weight_feature = self.fn(self.weight.view(self.weight.size(0), -1).t())
        #print(weight_feature[0:6, 0])
#         weight_feature = round_func_BPDA(2**(0.125*self.weight.view(self.weight.size(0), -1).t()) - 2**(-0.125*self.weight.view(self.weight.size(0), -1).t()))
#         weight_feature = (torch.ones(self.weight.shape) - 2**self.weight).view(self.weight.size(0), -1).t()
        # print(inp.shape)
        # print(in_feature.shape)
#         print(weight_feature)
#         out_unf = torch.matmul(inp_unf.transpose(1, 2), self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2)
        out_unf = torch.matmul(in_feature, weight_feature.unsqueeze(0))
#         print(out_unf.shape)
#         out_unf_check = matmul_cust(in_feature, weight_feature)

        # print(out_unf_check)
        # compare = out_unf_check - out_unf
        # print(compare)
        # print(compare.shape)
        #out_ = out_unf.view(batch_size, self.out_c, h_out, w_out) # this is equivalent with fold.
#         print(stride)
        out_ = torch.nn.functional.fold(torch.transpose(out_unf, 1, 2), (h_out, w_out), (1, 1))
        return out_


class QuantConv2d_Original(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(QuantConv2d_Original, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias)
        self.layer_type = 'QuantConv2d'
        self.bit = 4
        self.weight_quant = weight_quantize_fn(w_bit=self.bit, power=True)
        self.act_grid = build_POT_value(self.bit, additive=True)
        self.act_alq = act_quantization(self.bit, self.act_grid, power=True)
        self.act_alpha = torch.nn.Parameter(torch.tensor(8.0))
        self.fn = F_alpha.apply

    def forward(self, x):
        #weight_q = self.weight
        weight_q = self.fn(self.act_alpha, self.weight)
        
        #weight_q = self.weight_quant(self.weight)
        #print(weight_q[0, 0, :, :])
        #print('weight alpha', self.weight_quant.wgt_alpha)
        x = self.act_alq(x, self.act_alpha)
        x = F.conv2d(x, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

    def show_params(self):
        wgt_alpha = round(self.weight_quant.wgt_alpha.data.item(), 3)
        act_alpha = round(self.act_alpha.data.item(), 3)
        print('clipping threshold weight alpha: {:2f}, activation alpha: {:2f}'.format(wgt_alpha, act_alpha))
        weight_q = self.weight
        
        
class QuantLinear2d(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=False):
        super(QuantLinear2d, self).__init__(in_channels, out_channels,
                                          bias)
        self.layer_type = 'QuantLinear2d'
        self.bit = 4
        self.weight_quant = weight_quantize_fn(w_bit=self.bit, power=True)
        self.act_grid = build_POT_value(self.bit, additive=True)
        self.act_alq = act_quantization(self.bit, self.act_grid, power=True)
        self.act_alpha = torch.nn.Parameter(torch.tensor(8.0))

    def forward(self, x):
        weight_q = self.weight_quant(self.weight)
        x = self.act_alq(x, self.act_alpha)
        return F.linear(x, weight_q, self.bias)

    def show_params(self):
        wgt_alpha = round(self.weight_quant.wgt_alpha.data.item(), 3)
        act_alpha = round(self.act_alpha.data.item(), 3)
        print('clipping threshold weight alpha: {:2f}, activation alpha: {:2f}'.format(wgt_alpha, act_alpha))


# 8-bit quantization for the first and the last layer
class first_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(first_conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.layer_type = 'FConv2d'

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q-self.weight).detach()+self.weight
        return F.conv2d(x, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class last_fc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(last_fc, self).__init__(in_features, out_features, bias)
        self.layer_type = 'LFC'

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q-self.weight).detach()+self.weight
        return F.linear(x, weight_q, self.bias)
