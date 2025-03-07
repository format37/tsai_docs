## On this page

  * ResNetPlus
  * ResBlockPlus



  * __Report an issue



  1. Models
  2. CNNs
  3. ResNetPlus



# ResNetPlus

> This is an unofficial PyTorch implementation created by Ignacio Oguiza - oguiza@timeseriesAI.co

* * *

source

### ResNetPlus

> 
>      ResNetPlus (c_in, c_out, seq_len=None, nf=64, sa=False, se=None,
>                  fc_dropout=0.0, concat_pool=False, flatten=False,
>                  custom_head=None, y_range=None, ks=[7, 5, 3], coord=False,
>                  separable=False, bn_1st=True, zero_norm=False, act=<class
>                  'torch.nn.modules.activation.ReLU'>, act_kwargs={})

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

* * *

source

### ResBlockPlus

> 
>      ResBlockPlus (ni, nf, ks=[7, 5, 3], coord=False, separable=False,
>                    bn_1st=True, zero_norm=False, sa=False, se=None, act=<class
>                    'torch.nn.modules.activation.ReLU'>, act_kwargs={})

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    from tsai.models.layers import Swish __
    
    
    xb = torch.rand(2, 3, 4)
    test_eq(ResNetPlus(3,2)(xb).shape, [xb.shape[0], 2])
    test_eq(ResNetPlus(3,2,coord=True, separable=True, bn_1st=False, zero_norm=True, act=Swish, act_kwargs={}, fc_dropout=0.5)(xb).shape, [xb.shape[0], 2])
    test_eq(count_parameters(ResNetPlus(3, 2)), 479490) # for (3,2)__
    
    
    from tsai.models.ResNet import *__
    
    
    test_eq(count_parameters(ResNet(3, 2)), count_parameters(ResNetPlus(3, 2))) # for (3,2)__
    
    
    m = ResNetPlus(3, 2, zero_norm=True, coord=True, separable=True)
    print('n_params:', count_parameters(m))
    print(m)
    print(check_weight(m, is_bn)[0])__
    
    
    n_params: 114820
    ResNetPlus(
      (backbone): Sequential(
        (0): ResBlockPlus(
          (convblock1): ConvBlock(
            (0): AddCoords1d()
            (1): SeparableConv1d(
              (depthwise_conv): Conv1d(4, 4, kernel_size=(7,), stride=(1,), padding=(3,), groups=4, bias=False)
              (pointwise_conv): Conv1d(4, 64, kernel_size=(1,), stride=(1,), bias=False)
            )
            (2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (3): ReLU()
          )
          (convblock2): ConvBlock(
            (0): AddCoords1d()
            (1): SeparableConv1d(
              (depthwise_conv): Conv1d(65, 65, kernel_size=(5,), stride=(1,), padding=(2,), groups=65, bias=False)
              (pointwise_conv): Conv1d(65, 64, kernel_size=(1,), stride=(1,), bias=False)
            )
            (2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (3): ReLU()
          )
          (convblock3): ConvBlock(
            (0): AddCoords1d()
            (1): SeparableConv1d(
              (depthwise_conv): Conv1d(65, 65, kernel_size=(3,), stride=(1,), padding=(1,), groups=65, bias=False)
              (pointwise_conv): Conv1d(65, 64, kernel_size=(1,), stride=(1,), bias=False)
            )
            (2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): ConvBlock(
            (0): AddCoords1d()
            (1): Conv1d(4, 64, kernel_size=(1,), stride=(1,), bias=False)
            (2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (add): Add
          (act): ReLU()
        )
        (1): ResBlockPlus(
          (convblock1): ConvBlock(
            (0): AddCoords1d()
            (1): SeparableConv1d(
              (depthwise_conv): Conv1d(65, 65, kernel_size=(7,), stride=(1,), padding=(3,), groups=65, bias=False)
              (pointwise_conv): Conv1d(65, 128, kernel_size=(1,), stride=(1,), bias=False)
            )
            (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (3): ReLU()
          )
          (convblock2): ConvBlock(
            (0): AddCoords1d()
            (1): SeparableConv1d(
              (depthwise_conv): Conv1d(129, 129, kernel_size=(5,), stride=(1,), padding=(2,), groups=129, bias=False)
              (pointwise_conv): Conv1d(129, 128, kernel_size=(1,), stride=(1,), bias=False)
            )
            (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (3): ReLU()
          )
          (convblock3): ConvBlock(
            (0): AddCoords1d()
            (1): SeparableConv1d(
              (depthwise_conv): Conv1d(129, 129, kernel_size=(3,), stride=(1,), padding=(1,), groups=129, bias=False)
              (pointwise_conv): Conv1d(129, 128, kernel_size=(1,), stride=(1,), bias=False)
            )
            (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): ConvBlock(
            (0): AddCoords1d()
            (1): Conv1d(65, 128, kernel_size=(1,), stride=(1,), bias=False)
            (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (add): Add
          (act): ReLU()
        )
        (2): ResBlockPlus(
          (convblock1): ConvBlock(
            (0): AddCoords1d()
            (1): SeparableConv1d(
              (depthwise_conv): Conv1d(129, 129, kernel_size=(7,), stride=(1,), padding=(3,), groups=129, bias=False)
              (pointwise_conv): Conv1d(129, 128, kernel_size=(1,), stride=(1,), bias=False)
            )
            (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (3): ReLU()
          )
          (convblock2): ConvBlock(
            (0): AddCoords1d()
            (1): SeparableConv1d(
              (depthwise_conv): Conv1d(129, 129, kernel_size=(5,), stride=(1,), padding=(2,), groups=129, bias=False)
              (pointwise_conv): Conv1d(129, 128, kernel_size=(1,), stride=(1,), bias=False)
            )
            (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (3): ReLU()
          )
          (convblock3): ConvBlock(
            (0): AddCoords1d()
            (1): SeparableConv1d(
              (depthwise_conv): Conv1d(129, 129, kernel_size=(3,), stride=(1,), padding=(1,), groups=129, bias=False)
              (pointwise_conv): Conv1d(129, 128, kernel_size=(1,), stride=(1,), bias=False)
            )
            (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (add): Add
          (act): ReLU()
        )
      )
      (head): Sequential(
        (0): GAP1d(
          (gap): AdaptiveAvgPool1d(output_size=1)
          (flatten): Reshape(bs)
        )
        (1): Linear(in_features=128, out_features=2, bias=True)
      )
    )
    [1. 1. 0. 1. 1. 1. 0. 1. 1. 1. 0. 1.]

  * __Report an issue


