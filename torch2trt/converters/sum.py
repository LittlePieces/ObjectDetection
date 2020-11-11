from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
from .unary import UnaryModule
    

@tensorrt_converter('torch.sum')
@tensorrt_converter('torch.Tensor.sum')
def convert_sum(ctx):
    input = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=tuple(range(1, input.ndim)))
    keepdim = get_arg(ctx, 'keepdim', pos=2, default=False)
    input_trt= trt_(ctx.network, input)
    output = ctx.method_return
    layer = ctx.network.add_reduce(input_trt,  trt.ReduceOperation.SUM, torch_dim_to_trt_axes(dim), keepdim)
    output._trt = layer.get_output(0)
        
        
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_sum_reduce_all():
    return UnaryModule(lambda x: torch.sum(x))     


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_sum_reduce_dim1():
    return UnaryModule(lambda x: torch.sum(x, 1))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_sum_reduce_dim22():
    return UnaryModule(lambda x: torch.sum(x, 2))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_sum_reduce_dim1_keepdim():
    return UnaryModule(lambda x: torch.sum(x, 1, keepdim=True))