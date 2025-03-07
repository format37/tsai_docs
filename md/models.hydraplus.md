## On this page

  * HydraBackbonePlus
  * HydraPlus



  * __Report an issue



  1. Models
  2. ROCKETs
  3. HydraPlus



# HydraPlus

> Hydra: competing convolutional kernels for fast and accurate time series classification.

This is a Pytorch implementation of Hydra adapted by Ignacio Oguiza and based on:

Dempster, A., Schmidt, D. F., & Webb, G. I. (2023). Hydra: Competing convolutional kernels for fast and accurate time series classification. Data Mining and Knowledge Discovery, 1-27.

Original paper: https://link.springer.com/article/10.1007/s10618-023-00939-3

Original repository: https://github.com/angus924/hydra

* * *

source

### HydraBackbonePlus

> 
>      HydraBackbonePlus (c_in, c_out, seq_len, k=8, g=64, max_c_in=8,
>                         clip=True, device=device(type='cpu'), zero_init=True)

*Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes::
    
    
    import torch.nn as nn
    import torch.nn.functional as F
    
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)
    
        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will have their parameters converted too when you call :meth:`to`, etc.

.. note:: As per the example above, an `__init__()` call to the parent class must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or evaluation mode. :vartype training: bool*

* * *

source

### HydraPlus

> 
>      HydraPlus (c_in:int, c_out:int, seq_len:int, d:tuple=None, k:int=8,
>                 g:int=64, max_c_in:int=8, clip:bool=True, use_bn:bool=True,
>                 fc_dropout:float=0.0, custom_head:Any=None,
>                 zero_init:bool=True, use_diff:bool=True,
>                 device:str=device(type='cpu'))

*A sequential container.

Modules will be added to it in the order they are passed in the constructor. Alternatively, an `OrderedDict` of modules can be passed in. The `forward()` method of `[`Sequential`](https://timeseriesAI.github.io/models.layers.html#sequential)` accepts any input and forwards it to the first module it contains. It then “chains” outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.

The value a `[`Sequential`](https://timeseriesAI.github.io/models.layers.html#sequential)` provides over manually calling a sequence of modules is that it allows treating the whole container as a single module, such that performing a transformation on the `[`Sequential`](https://timeseriesAI.github.io/models.layers.html#sequential)` applies to each of the modules it stores (which are each a registered submodule of the `[`Sequential`](https://timeseriesAI.github.io/models.layers.html#sequential)`).

What’s the difference between a `[`Sequential`](https://timeseriesAI.github.io/models.layers.html#sequential)` and a :class:`torch.nn.ModuleList`? A `ModuleList` is exactly what it sounds like–a list for storing `Module` s! On the other hand, the layers in a `[`Sequential`](https://timeseriesAI.github.io/models.layers.html#sequential)` are connected in a cascading way.

Example::
    
    
    # Using Sequential to create a small model. When `model` is run,
    # input will first be passed to `Conv2d(1,20,5)`. The output of
    # `Conv2d(1,20,5)` will be used as the input to the first
    # `ReLU`; the output of the first `ReLU` will become the input
    # for `Conv2d(20,64,5)`. Finally, the output of
    # `Conv2d(20,64,5)` will be used as input to the second `ReLU`
    model = nn.Sequential(
              nn.Conv2d(1,20,5),
              nn.ReLU(),
              nn.Conv2d(20,64,5),
              nn.ReLU()
            )
    
    # Using Sequential with OrderedDict. This is functionally the
    # same as the above code
    model = nn.Sequential(OrderedDict([
              ('conv1', nn.Conv2d(1,20,5)),
              ('relu1', nn.ReLU()),
              ('conv2', nn.Conv2d(20,64,5)),
              ('relu2', nn.ReLU())
            ]))*

| **Type** | **Default** | **Details**  
---|---|---|---  
c_in | int |  | num of channels in input  
c_out | int |  | num of channels in output  
seq_len | int |  | sequence length  
d | tuple | None | shape of the output (when ndim > 1)  
k | int | 8 | number of kernels per group  
g | int | 64 | number of groups  
max_c_in | int | 8 | max number of channels per group  
clip | bool | True | clip values >= 0  
use_bn | bool | True | use batch norm  
fc_dropout | float | 0.0 | dropout probability  
custom_head | Any | None | optional custom head as a torch.nn.Module or Callable  
zero_init | bool | True | set head weights and biases to zero  
use_diff | bool | True | use diff(X) as input  
device | str | cpu | device to use  
      
    
    xb = torch.randn(16, 5, 20).to(default_device())
    yb = torch.randint(0, 3, (16, 20)).to(default_device())
    
    model = HydraPlus(5, 3, 20, d=None).to(default_device())
    output = model(xb)
    assert output.shape == (16, 3)
    output.shape __
    
    
    torch.Size([16, 3])
    
    
    xb = torch.randn(16, 5, 20).to(default_device())
    yb = torch.randint(0, 3, (16, 20)).to(default_device())
    
    model = HydraPlus(5, 3, 20, d=None, use_diff=False).to(default_device())
    output = model(xb)
    assert output.shape == (16, 3)
    output.shape __
    
    
    torch.Size([16, 3])
    
    
    xb = torch.randn(16, 5, 20).to(default_device())
    yb = torch.randint(0, 3, (16, 5, 20)).to(default_device())
    
    model = HydraPlus(5, 3, 20, d=20, use_diff=True).to(default_device())
    output = model(xb)
    assert output.shape == (16, 20, 3)
    output.shape __
    
    
    torch.Size([16, 20, 3])

  * __Report an issue


