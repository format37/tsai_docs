## On this page

  * FCNPlus



  * __Report an issue



  1. Models
  2. CNNs
  3. FCNPlus



# FCNPlus

> This is an unofficial PyTorch implementation created by Ignacio Oguiza - oguiza@timeseriesAI.co

* * *

source

### FCNPlus

> 
>      FCNPlus (c_in, c_out, layers=[128, 256, 128], kss=[7, 5, 3], coord=False,
>               separable=False, use_bn=False, fc_dropout=0.0, zero_norm=False,
>               act=<class 'torch.nn.modules.activation.ReLU'>, act_kwargs={},
>               residual=False, custom_head=None)

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
    
    
    xb = torch.rand(16, 3, 10)
    test_eq(FCNPlus(3, 2)(xb).shape, [xb.shape[0], 2])
    test_eq(FCNPlus(3, 2, coord=True, separable=True, act=Swish, residual=True)(xb).shape, [xb.shape[0], 2])
    test_eq(nn.Sequential(*FCNPlus(3, 2).children())(xb).shape, [xb.shape[0], 2])
    test_eq(FCNPlus(3, 2, custom_head=partial(mlp_head, seq_len=10))(xb).shape, [xb.shape[0], 2])__
    
    
    from tsai.models.utils import *__
    
    
    model = build_ts_model(FCNPlus, 2, 3)
    model[-1]__
    
    
    Sequential(
      (0): AdaptiveAvgPool1d(output_size=1)
      (1): Squeeze(dim=-1)
      (2): Linear(in_features=128, out_features=3, bias=True)
    )
    
    
    from tsai.models.FCN import *__
    
    
    test_eq(count_parameters(FCN(3,2)), count_parameters(FCNPlus(3,2)))__
    
    
    FCNPlus(3,2)__
    
    
    FCNPlus(
      (backbone): _FCNBlockPlus(
        (convblock1): ConvBlock(
          (0): Conv1d(3, 128, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
          (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (convblock2): ConvBlock(
          (0): Conv1d(128, 256, kernel_size=(5,), stride=(1,), padding=(2,), bias=False)
          (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (convblock3): ConvBlock(
          (0): Conv1d(256, 128, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
          (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (add): Sequential()
      )
      (head): Sequential(
        (0): AdaptiveAvgPool1d(output_size=1)
        (1): Squeeze(dim=-1)
        (2): Linear(in_features=128, out_features=2, bias=True)
      )
    )

  * __Report an issue


