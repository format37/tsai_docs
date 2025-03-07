## On this page

  * XceptionTimePlus
  * XceptionBlockPlus
  * XceptionModulePlus



  * __Report an issue



  1. Models
  2. CNNs
  3. XceptionTimePlus



# XceptionTimePlus

This is an unofficial PyTorch implementation by Ignacio Oguiza - oguiza@timeseriesAI.co modified on:

Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J. & Petitjean, F. (2019). **InceptionTime: Finding AlexNet for Time Series Classification**. arXiv preprint arXiv:1909.04939.

Official InceptionTime tensorflow implementation: https://github.com/hfawaz/InceptionTime

* * *

source

### XceptionTimePlus

> 
>      XceptionTimePlus (c_in, c_out, seq_len=None, nf=16, nb_filters=None,
>                        coord=False, norm='Batch', concat_pool=False,
>                        adaptive_size=50, custom_head=None, residual=True,
>                        zero_norm=False, act=<class
>                        'torch.nn.modules.activation.ReLU'>, act_kwargs={})

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

### XceptionBlockPlus

> 
>      XceptionBlockPlus (ni, nf, residual=True, coord=False, norm='Batch',
>                         zero_norm=False, act=<class
>                         'torch.nn.modules.activation.ReLU'>, act_kwargs={},
>                         ks=40, kss=None, bottleneck=True, separable=True,
>                         bn_1st=True, norm_act=False)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### XceptionModulePlus

> 
>      XceptionModulePlus (ni, nf, ks=40, kss=None, bottleneck=True,
>                          coord=False, separable=True, norm='Batch',
>                          zero_norm=False, bn_1st=True, act=<class
>                          'torch.nn.modules.activation.ReLU'>, act_kwargs={},
>                          norm_act=False)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    bs = 16
    vars = 3
    seq_len = 12
    c_out = 2
    xb = torch.rand(bs, vars, seq_len)__
    
    
    test_eq(XceptionTimePlus(vars,c_out)(xb).shape, [bs, c_out])
    test_eq(XceptionTimePlus(vars,c_out, nf=32)(xb).shape, [bs, c_out])
    test_eq(XceptionTimePlus(vars,c_out, bottleneck=False)(xb).shape, [bs, c_out])
    test_eq(XceptionTimePlus(vars,c_out, residual=False)(xb).shape, [bs, c_out])
    test_eq(XceptionTimePlus(vars,c_out, coord=True)(xb).shape, [bs, c_out])
    test_eq(XceptionTimePlus(vars,c_out, concat_pool=True)(xb).shape, [bs, c_out])
    test_eq(count_parameters(XceptionTimePlus(3, 2)), 399540)__
    
    
    m = XceptionTimePlus(2,3)
    test_eq(check_weight(m, is_bn)[0].sum(), 5)
    test_eq(len(check_bias(m, is_conv)[0]), 0)
    m = XceptionTimePlus(2,3, zero_norm=True)
    test_eq(check_weight(m, is_bn)[0].sum(), 5)
    m = XceptionTimePlus(2,3, zero_norm=True, norm_act=True)
    test_eq(check_weight(m, is_bn)[0].sum(), 7)__
    
    
    m = XceptionTimePlus(2,3, coord=True)
    test_eq(len(get_layers(m, cond=is_layer(AddCoords1d))), 25)
    test_eq(len(get_layers(m, cond=is_layer(nn.Conv1d))), 37)
    m = XceptionTimePlus(2,3, bottleneck=False, coord=True)
    test_eq(len(get_layers(m, cond=is_layer(AddCoords1d))), 21)
    test_eq(len(get_layers(m, cond=is_layer(nn.Conv1d))), 33)__
    
    
    m = XceptionTimePlus(vars, c_out, seq_len=seq_len, custom_head=mlp_head)
    test_eq(m(xb).shape, [bs, c_out])__
    
    
    XceptionTimePlus(vars, c_out, coord=True)__
    
    
    XceptionTimePlus(
      (backbone): XceptionBlockPlus(
        (xception): ModuleList(
          (0): XceptionModulePlus(
            (bottleneck): ConvBlock(
              (0): AddCoords1d()
              (1): Conv1d(4, 16, kernel_size=(1,), stride=(1,), bias=False)
            )
            (convs): ModuleList(
              (0): ConvBlock(
                (0): AddCoords1d()
                (1): SeparableConv1d(
                  (depthwise_conv): Conv1d(17, 17, kernel_size=(39,), stride=(1,), padding=(19,), groups=17, bias=False)
                  (pointwise_conv): Conv1d(17, 16, kernel_size=(1,), stride=(1,), bias=False)
                )
              )
              (1): ConvBlock(
                (0): AddCoords1d()
                (1): SeparableConv1d(
                  (depthwise_conv): Conv1d(17, 17, kernel_size=(19,), stride=(1,), padding=(9,), groups=17, bias=False)
                  (pointwise_conv): Conv1d(17, 16, kernel_size=(1,), stride=(1,), bias=False)
                )
              )
              (2): ConvBlock(
                (0): AddCoords1d()
                (1): SeparableConv1d(
                  (depthwise_conv): Conv1d(17, 17, kernel_size=(9,), stride=(1,), padding=(4,), groups=17, bias=False)
                  (pointwise_conv): Conv1d(17, 16, kernel_size=(1,), stride=(1,), bias=False)
                )
              )
            )
            (mp_conv): Sequential(
              (0): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
              (1): ConvBlock(
                (0): AddCoords1d()
                (1): Conv1d(4, 16, kernel_size=(1,), stride=(1,), bias=False)
              )
            )
            (concat): Concat(dim=1)
          )
          (1): XceptionModulePlus(
            (bottleneck): ConvBlock(
              (0): AddCoords1d()
              (1): Conv1d(65, 32, kernel_size=(1,), stride=(1,), bias=False)
            )
            (convs): ModuleList(
              (0): ConvBlock(
                (0): AddCoords1d()
                (1): SeparableConv1d(
                  (depthwise_conv): Conv1d(33, 33, kernel_size=(39,), stride=(1,), padding=(19,), groups=33, bias=False)
                  (pointwise_conv): Conv1d(33, 32, kernel_size=(1,), stride=(1,), bias=False)
                )
              )
              (1): ConvBlock(
                (0): AddCoords1d()
                (1): SeparableConv1d(
                  (depthwise_conv): Conv1d(33, 33, kernel_size=(19,), stride=(1,), padding=(9,), groups=33, bias=False)
                  (pointwise_conv): Conv1d(33, 32, kernel_size=(1,), stride=(1,), bias=False)
                )
              )
              (2): ConvBlock(
                (0): AddCoords1d()
                (1): SeparableConv1d(
                  (depthwise_conv): Conv1d(33, 33, kernel_size=(9,), stride=(1,), padding=(4,), groups=33, bias=False)
                  (pointwise_conv): Conv1d(33, 32, kernel_size=(1,), stride=(1,), bias=False)
                )
              )
            )
            (mp_conv): Sequential(
              (0): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
              (1): ConvBlock(
                (0): AddCoords1d()
                (1): Conv1d(65, 32, kernel_size=(1,), stride=(1,), bias=False)
              )
            )
            (concat): Concat(dim=1)
          )
          (2): XceptionModulePlus(
            (bottleneck): ConvBlock(
              (0): AddCoords1d()
              (1): Conv1d(129, 64, kernel_size=(1,), stride=(1,), bias=False)
            )
            (convs): ModuleList(
              (0): ConvBlock(
                (0): AddCoords1d()
                (1): SeparableConv1d(
                  (depthwise_conv): Conv1d(65, 65, kernel_size=(39,), stride=(1,), padding=(19,), groups=65, bias=False)
                  (pointwise_conv): Conv1d(65, 64, kernel_size=(1,), stride=(1,), bias=False)
                )
              )
              (1): ConvBlock(
                (0): AddCoords1d()
                (1): SeparableConv1d(
                  (depthwise_conv): Conv1d(65, 65, kernel_size=(19,), stride=(1,), padding=(9,), groups=65, bias=False)
                  (pointwise_conv): Conv1d(65, 64, kernel_size=(1,), stride=(1,), bias=False)
                )
              )
              (2): ConvBlock(
                (0): AddCoords1d()
                (1): SeparableConv1d(
                  (depthwise_conv): Conv1d(65, 65, kernel_size=(9,), stride=(1,), padding=(4,), groups=65, bias=False)
                  (pointwise_conv): Conv1d(65, 64, kernel_size=(1,), stride=(1,), bias=False)
                )
              )
            )
            (mp_conv): Sequential(
              (0): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
              (1): ConvBlock(
                (0): AddCoords1d()
                (1): Conv1d(129, 64, kernel_size=(1,), stride=(1,), bias=False)
              )
            )
            (concat): Concat(dim=1)
          )
          (3): XceptionModulePlus(
            (bottleneck): ConvBlock(
              (0): AddCoords1d()
              (1): Conv1d(257, 128, kernel_size=(1,), stride=(1,), bias=False)
            )
            (convs): ModuleList(
              (0): ConvBlock(
                (0): AddCoords1d()
                (1): SeparableConv1d(
                  (depthwise_conv): Conv1d(129, 129, kernel_size=(39,), stride=(1,), padding=(19,), groups=129, bias=False)
                  (pointwise_conv): Conv1d(129, 128, kernel_size=(1,), stride=(1,), bias=False)
                )
              )
              (1): ConvBlock(
                (0): AddCoords1d()
                (1): SeparableConv1d(
                  (depthwise_conv): Conv1d(129, 129, kernel_size=(19,), stride=(1,), padding=(9,), groups=129, bias=False)
                  (pointwise_conv): Conv1d(129, 128, kernel_size=(1,), stride=(1,), bias=False)
                )
              )
              (2): ConvBlock(
                (0): AddCoords1d()
                (1): SeparableConv1d(
                  (depthwise_conv): Conv1d(129, 129, kernel_size=(9,), stride=(1,), padding=(4,), groups=129, bias=False)
                  (pointwise_conv): Conv1d(129, 128, kernel_size=(1,), stride=(1,), bias=False)
                )
              )
            )
            (mp_conv): Sequential(
              (0): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
              (1): ConvBlock(
                (0): AddCoords1d()
                (1): Conv1d(257, 128, kernel_size=(1,), stride=(1,), bias=False)
              )
            )
            (concat): Concat(dim=1)
          )
        )
        (shortcut): ModuleList(
          (0): ConvBlock(
            (0): AddCoords1d()
            (1): Conv1d(4, 128, kernel_size=(1,), stride=(1,), bias=False)
            (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (0): AddCoords1d()
            (1): Conv1d(129, 512, kernel_size=(1,), stride=(1,), bias=False)
            (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (act): ModuleList(
          (0): ReLU()
          (1): ReLU()
        )
        (add): Add
      )
      (head): Sequential(
        (0): AdaptiveAvgPool1d(output_size=50)
        (1): ConvBlock(
          (0): AddCoords1d()
          (1): Conv1d(513, 256, kernel_size=(1,), stride=(1,), bias=False)
          (2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU()
        )
        (2): ConvBlock(
          (0): AddCoords1d()
          (1): Conv1d(257, 128, kernel_size=(1,), stride=(1,), bias=False)
          (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU()
        )
        (3): ConvBlock(
          (0): AddCoords1d()
          (1): Conv1d(129, 2, kernel_size=(1,), stride=(1,), bias=False)
          (2): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU()
        )
        (4): GAP1d(
          (gap): AdaptiveAvgPool1d(output_size=1)
          (flatten): Reshape(bs)
        )
      )
    )

  * __Report an issue


