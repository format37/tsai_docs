## On this page

  * Flatten
  * MultiRocketFeaturesPlus
  * MultiRocketBackbonePlus
  * MultiRocketPlus



  * __Report an issue



  1. Models
  2. ROCKETs
  3. MultiRocketPlus



# MultiRocketPlus

> MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification.

This is a Pytorch implementation of MultiRocket developed by Malcolm McLean and Ignacio Oguiza based on:

Tan, C. W., Dempster, A., Bergmeir, C., & Webb, G. I. (2022). MultiRocket: multiple pooling operators and transformations for fast and effective time series classification. Data Mining and Knowledge Discovery, 36(5), 1623-1646.

Original paper: https://link.springer.com/article/10.1007/s10618-022-00844-1

Original repository: https://github.com/ChangWeiTan/MultiRocket

* * *

source

### Flatten

> 
>      Flatten (*args, **kwargs)

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
    
    
    from tsai.imports import default_device __
    
    
    o = torch.rand(2, 3, 5, 4).to(default_device()) - .3
    print(o)
    
    output = _LPVV(o, dim=2)
    print(output)  # Should print: torch.Size([2, 3, 4])__
    
    
    tensor([[[[ 0.5644, -0.0509, -0.0390,  0.4091],
              [ 0.0517, -0.1471,  0.6458,  0.5593],
              [ 0.4516, -0.0821,  0.1271,  0.0592],
              [ 0.4151,  0.4376,  0.0763,  0.3780],
              [ 0.2653, -0.1817,  0.0156,  0.4993]],
    
             [[-0.0779,  0.0858,  0.1982,  0.3224],
              [ 0.1130,  0.0714, -0.1779,  0.5360],
              [-0.1848, -0.2270, -0.0925, -0.1217],
              [ 0.2820, -0.0205, -0.2777,  0.3755],
              [-0.2490,  0.2613,  0.4237,  0.4534]],
    
             [[-0.0162,  0.6368,  0.0016,  0.1467],
              [ 0.6035, -0.1365,  0.6930,  0.6943],
              [ 0.2790,  0.3818, -0.0731,  0.0167],
              [ 0.6442,  0.3443,  0.4829, -0.0944],
              [ 0.2932,  0.6952,  0.5541,  0.5946]]],
    
    
            [[[ 0.6757,  0.5740,  0.3071,  0.4400],
              [-0.2344, -0.1056,  0.4773,  0.2432],
              [ 0.2595, -0.1528, -0.0866,  0.6201],
              [ 0.0657,  0.1220,  0.4849,  0.4254],
              [ 0.3399, -0.1609,  0.3465,  0.2389]],
    
             [[-0.0765,  0.0516,  0.0028,  0.4381],
              [ 0.5212, -0.2781, -0.0896, -0.0301],
              [ 0.6857,  0.3583,  0.5869,  0.3418],
              [ 0.3002,  0.5135,  0.6011,  0.6499],
              [-0.2807, -0.2888,  0.3965,  0.6585]],
    
             [[-0.1368,  0.6677,  0.1439,  0.1434],
              [-0.1820,  0.1041, -0.1211,  0.6103],
              [ 0.5808,  0.4588,  0.4572,  0.3713],
              [ 0.2389, -0.1392,  0.1371, -0.1570],
              [ 0.2840,  0.1214, -0.0059,  0.5064]]]], device='mps:0')
    tensor([[[ 1.0000, -0.6000,  0.6000,  1.0000],
             [-0.6000, -0.2000, -0.6000, -0.2000],
             [ 0.6000,  0.2000, -0.2000,  0.2000]],
    
            [[ 0.2000, -0.6000, -0.2000,  1.0000],
             [ 0.2000, -0.2000,  0.2000,  0.2000],
             [ 0.2000,  0.2000, -0.2000,  0.2000]]], device='mps:0')
    
    
    output = _MPV(o, dim=2)
    print(output)  # Should print: torch.Size([2, 3, 4])__
    
    
    tensor([[[0.3496, 0.4376, 0.2162, 0.3810],
             [0.1975, 0.1395, 0.3109, 0.4218],
             [0.4550, 0.5145, 0.4329, 0.3631]],
    
            [[0.3352, 0.3480, 0.4040, 0.3935],
             [0.5023, 0.3078, 0.3968, 0.5221],
             [0.3679, 0.3380, 0.2460, 0.4079]]], device='mps:0')
    
    
    output = _RSPV(o, dim=2)
    print(output)  # Should print: torch.Size([2, 3, 4])__
    
    
    tensor([[[ 1.0000, -0.0270,  0.9138,  1.0000],
             [-0.1286,  0.2568,  0.0630,  0.8654],
             [ 0.9823,  0.8756,  0.9190,  0.8779]],
    
            [[ 0.7024,  0.2482,  0.8983,  1.0000],
             [ 0.6168,  0.2392,  0.8931,  0.9715],
             [ 0.5517,  0.8133,  0.7065,  0.8244]]], device='mps:0')
    
    
    output = _PPV(o, dim=2)
    print(output)  # Should print: torch.Size([2, 3, 4])__
    
    
    tensor([[[-0.3007, -1.0097, -0.6697, -0.2381],
             [-1.0466, -0.9316, -0.9705, -0.3738],
             [-0.2786, -0.2314, -0.3366, -0.4569]],
    
            [[-0.5574, -0.8893, -0.3883, -0.2130],
             [-0.5401, -0.8574, -0.4009, -0.1767],
             [-0.6861, -0.5149, -0.7555, -0.4102]]], device='mps:0')

* * *

source

### MultiRocketFeaturesPlus

> 
>      MultiRocketFeaturesPlus (c_in, seq_len, num_features=10000,
>                               max_dilations_per_kernel=32, kernel_size=9,
>                               max_num_channels=9, max_num_kernels=84,
>                               diff=False)

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

### MultiRocketBackbonePlus

> 
>      MultiRocketBackbonePlus (c_in, seq_len, num_features=50000,
>                               max_dilations_per_kernel=32, kernel_size=9,
>                               max_num_channels=None, max_num_kernels=84,
>                               use_diff=True)

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

### MultiRocketPlus

> 
>      MultiRocketPlus (c_in, c_out, seq_len, d=None, num_features=50000,
>                       max_dilations_per_kernel=32, kernel_size=9,
>                       max_num_channels=None, max_num_kernels=84, use_bn=True,
>                       fc_dropout=0, custom_head=None, zero_init=True,
>                       use_diff=True)

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
    
    
    from tsai.imports import default_device __
    
    
    xb = torch.randn(16, 5, 20).to(default_device())
    yb = torch.randint(0, 3, (16, 20)).to(default_device())
    
    model = MultiRocketPlus(5, 3, 20, d=None, use_diff=True).to(default_device())
    output = model(xb)
    assert output.shape == (16, 3)
    output.shape __
    
    
    torch.Size([16, 3])
    
    
    xb = torch.randn(16, 5, 20).to(default_device())
    yb = torch.randint(0, 3, (16, 20)).to(default_device())
    
    model = MultiRocketPlus(5, 3, 20, d=None, use_diff=False).to(default_device())
    output = model(xb)
    assert output.shape == (16, 3)
    output.shape __
    
    
    torch.Size([16, 3])
    
    
    xb = torch.randn(16, 5, 20).to(default_device())
    yb = torch.randint(0, 3, (16, 5, 20)).to(default_device())
    
    model = MultiRocketPlus(5, 3, 20, d=20, use_diff=True).to(default_device())
    output = model(xb)
    assert output.shape == (16, 20, 3)
    output.shape __
    
    
    torch.Size([16, 20, 3])

  * __Report an issue


