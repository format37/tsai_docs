## On this page

  * InceptionBlockPlus
  * InceptionModulePlus
  * InceptionTimePlus
  * XCoordTime
  * InCoordTime
  * MultiInceptionTimePlus



  * __Report an issue



  1. Models
  2. CNNs
  3. InceptionTimePlus



# InceptionTimePlus

> This is an unofficial PyTorch implementation of InceptionTime (Fawaz, 2019) created by Ignacio Oguiza.

**References:** * Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J., … & Petitjean, F. (2020). Inceptiontime: Finding alexnet for time series classification. Data Mining and Knowledge Discovery, 34(6), 1936-1962. * Official InceptionTime tensorflow implementation: https://github.com/hfawaz/InceptionTime

* * *

source

### InceptionBlockPlus

> 
>      InceptionBlockPlus (ni, nf, residual=True, depth=6, coord=False,
>                          norm='Batch', zero_norm=False, act=<class
>                          'torch.nn.modules.activation.ReLU'>, act_kwargs={},
>                          sa=False, se=None, stoch_depth=1.0, ks=40,
>                          bottleneck=True, padding='same', separable=False,
>                          dilation=1, stride=1, conv_dropout=0.0, bn_1st=True)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### InceptionModulePlus

> 
>      InceptionModulePlus (ni, nf, ks=40, bottleneck=True, padding='same',
>                           coord=False, separable=False, dilation=1, stride=1,
>                           conv_dropout=0.0, sa=False, se=None, norm='Batch',
>                           zero_norm=False, bn_1st=True, act=<class
>                           'torch.nn.modules.activation.ReLU'>, act_kwargs={})

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### InceptionTimePlus

> 
>      InceptionTimePlus (c_in, c_out, seq_len=None, nf=32, nb_filters=None,
>                         flatten=False, concat_pool=False, fc_dropout=0.0,
>                         bn=False, y_range=None, custom_head=None, ks=40,
>                         bottleneck=True, padding='same', coord=False,
>                         separable=False, dilation=1, stride=1,
>                         conv_dropout=0.0, sa=False, se=None, norm='Batch',
>                         zero_norm=False, bn_1st=True, act=<class
>                         'torch.nn.modules.activation.ReLU'>, act_kwargs={})

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

### XCoordTime

> 
>      XCoordTime (c_in, c_out, seq_len=None, nf=32, nb_filters=None,
>                  flatten=False, concat_pool=False, fc_dropout=0.0, bn=False,
>                  y_range=None, custom_head=None, ks=40, bottleneck=True,
>                  padding='same', coord=False, separable=False, dilation=1,
>                  stride=1, conv_dropout=0.0, sa=False, se=None, norm='Batch',
>                  zero_norm=False, bn_1st=True, act=<class
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

### InCoordTime

> 
>      InCoordTime (c_in, c_out, seq_len=None, nf=32, nb_filters=None,
>                   flatten=False, concat_pool=False, fc_dropout=0.0, bn=False,
>                   y_range=None, custom_head=None, ks=40, bottleneck=True,
>                   padding='same', coord=False, separable=False, dilation=1,
>                   stride=1, conv_dropout=0.0, sa=False, se=None, norm='Batch',
>                   zero_norm=False, bn_1st=True, act=<class
>                   'torch.nn.modules.activation.ReLU'>, act_kwargs={})

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
    
    
    from tsai.data.core import TSCategorize
    from tsai.models.utils import count_parameters __
    
    
    bs = 16
    n_vars = 3
    seq_len = 51
    c_out = 2
    xb = torch.rand(bs, n_vars, seq_len)
    
    test_eq(InceptionTimePlus(n_vars,c_out)(xb).shape, [bs, c_out])
    test_eq(InceptionTimePlus(n_vars,c_out,concat_pool=True)(xb).shape, [bs, c_out])
    test_eq(InceptionTimePlus(n_vars,c_out, bottleneck=False)(xb).shape, [bs, c_out])
    test_eq(InceptionTimePlus(n_vars,c_out, residual=False)(xb).shape, [bs, c_out])
    test_eq(InceptionTimePlus(n_vars,c_out, conv_dropout=.5)(xb).shape, [bs, c_out])
    test_eq(InceptionTimePlus(n_vars,c_out, stoch_depth=.5)(xb).shape, [bs, c_out])
    test_eq(InceptionTimePlus(n_vars, c_out, seq_len=seq_len, zero_norm=True, flatten=True)(xb).shape, [bs, c_out])
    test_eq(InceptionTimePlus(n_vars,c_out, coord=True, separable=True, 
                              norm='Instance', zero_norm=True, bn_1st=False, fc_dropout=.5, sa=True, se=True, act=nn.PReLU, act_kwargs={})(xb).shape, [bs, c_out])
    test_eq(InceptionTimePlus(n_vars,c_out, coord=True, separable=True,
                              norm='Instance', zero_norm=True, bn_1st=False, act=nn.PReLU, act_kwargs={})(xb).shape, [bs, c_out])
    test_eq(count_parameters(InceptionTimePlus(3, 2)), 455490)
    test_eq(count_parameters(InceptionTimePlus(6, 2, **{'coord': True, 'separable': True, 'zero_norm': True})), 77204)
    test_eq(count_parameters(InceptionTimePlus(3, 2, ks=40)), count_parameters(InceptionTimePlus(3, 2, ks=[9, 19, 39])))__
    
    
    bs = 16
    n_vars = 3
    seq_len = 51
    c_out = 2
    xb = torch.rand(bs, n_vars, seq_len)
    
    model = InceptionTimePlus(n_vars, c_out)
    model(xb).shape
    test_eq(model[0](xb), model.backbone(xb))
    test_eq(model[1](model[0](xb)), model.head(model[0](xb)))
    test_eq(model[1].state_dict().keys(), model.head.state_dict().keys())
    test_eq(len(ts_splitter(model)), 2)__
    
    
    test_eq(check_bias(InceptionTimePlus(2,3, zero_norm=True), is_conv)[0].sum(), 0)
    test_eq(check_weight(InceptionTimePlus(2,3, zero_norm=True), is_bn)[0].sum(), 6)
    test_eq(check_weight(InceptionTimePlus(2,3), is_bn)[0], np.array([1., 1., 1., 1., 1., 1., 1., 1.]))__
    
    
    for i in range(10): InceptionTimePlus(n_vars,c_out,stoch_depth=0.8,depth=9,zero_norm=True)(xb)__
    
    
    net = InceptionTimePlus(2,3,**{'coord': True, 'separable': True, 'zero_norm': True})
    test_eq(check_weight(net, is_bn)[0], np.array([1., 1., 0., 1., 1., 0., 1., 1.]))
    net __
    
    
    InceptionTimePlus(
      (backbone): Sequential(
        (0): InceptionBlockPlus(
          (inception): ModuleList(
            (0): InceptionModulePlus(
              (bottleneck): ConvBlock(
                (0): AddCoords1d()
                (1): Conv1d(3, 32, kernel_size=(1,), stride=(1,), bias=False)
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
                  (1): Conv1d(3, 32, kernel_size=(1,), stride=(1,), bias=False)
                )
              )
              (concat): Concat(dim=1)
              (norm): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): ReLU()
            )
            (1): InceptionModulePlus(
              (bottleneck): ConvBlock(
                (0): AddCoords1d()
                (1): Conv1d(129, 32, kernel_size=(1,), stride=(1,), bias=False)
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
                  (1): Conv1d(129, 32, kernel_size=(1,), stride=(1,), bias=False)
                )
              )
              (concat): Concat(dim=1)
              (norm): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): ReLU()
            )
            (2): InceptionModulePlus(
              (bottleneck): ConvBlock(
                (0): AddCoords1d()
                (1): Conv1d(129, 32, kernel_size=(1,), stride=(1,), bias=False)
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
                  (1): Conv1d(129, 32, kernel_size=(1,), stride=(1,), bias=False)
                )
              )
              (concat): Concat(dim=1)
              (norm): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (3): InceptionModulePlus(
              (bottleneck): ConvBlock(
                (0): AddCoords1d()
                (1): Conv1d(129, 32, kernel_size=(1,), stride=(1,), bias=False)
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
                  (1): Conv1d(129, 32, kernel_size=(1,), stride=(1,), bias=False)
                )
              )
              (concat): Concat(dim=1)
              (norm): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): ReLU()
            )
            (4): InceptionModulePlus(
              (bottleneck): ConvBlock(
                (0): AddCoords1d()
                (1): Conv1d(129, 32, kernel_size=(1,), stride=(1,), bias=False)
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
                  (1): Conv1d(129, 32, kernel_size=(1,), stride=(1,), bias=False)
                )
              )
              (concat): Concat(dim=1)
              (norm): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): ReLU()
            )
            (5): InceptionModulePlus(
              (bottleneck): ConvBlock(
                (0): AddCoords1d()
                (1): Conv1d(129, 32, kernel_size=(1,), stride=(1,), bias=False)
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
                  (1): Conv1d(129, 32, kernel_size=(1,), stride=(1,), bias=False)
                )
              )
              (concat): Concat(dim=1)
              (norm): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (shortcut): ModuleList(
            (0): ConvBlock(
              (0): AddCoords1d()
              (1): Conv1d(3, 128, kernel_size=(1,), stride=(1,), bias=False)
              (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (act): ModuleList(
            (0): ReLU()
            (1): ReLU()
          )
          (add): Add
        )
      )
      (head): Sequential(
        (0): Sequential(
          (0): GAP1d(
            (gap): AdaptiveAvgPool1d(output_size=1)
            (flatten): Reshape(bs)
          )
          (1): LinBnDrop(
            (0): Linear(in_features=128, out_features=3, bias=True)
          )
        )
      )
    )

* * *

source

### MultiInceptionTimePlus

> 
>      MultiInceptionTimePlus (feat_list, c_out, seq_len=None, nf=32,
>                              nb_filters=None, depth=6, stoch_depth=1.0,
>                              flatten=False, concat_pool=False, fc_dropout=0.0,
>                              bn=False, y_range=None, custom_head=None)

_Class that allows you to create a model with multiple branches of InceptionTimePlus._
    
    
    bs = 16
    n_vars = 3
    seq_len = 51
    c_out = 2
    xb = torch.rand(bs, n_vars, seq_len)
    
    test_eq(count_parameters(MultiInceptionTimePlus([1,1,1], c_out)) > count_parameters(MultiInceptionTimePlus(3, c_out)), True)
    test_eq(MultiInceptionTimePlus([1,1,1], c_out).to(xb.device)(xb).shape, MultiInceptionTimePlus(3, c_out).to(xb.device)(xb).shape)__
    
    
    [W NNPACK.cpp:53] Could not initialize NNPACK! Reason: Unsupported hardware.
    
    
    bs = 16
    n_vars = 3
    seq_len = 12
    c_out = 10
    xb = torch.rand(bs, n_vars, seq_len)
    new_head = partial(conv_lin_nd_head, d=(5,2))
    net = MultiInceptionTimePlus(n_vars, c_out, seq_len, custom_head=new_head)
    print(net.to(xb.device)(xb).shape)
    net.head __
    
    
    torch.Size([16, 5, 2, 10])
    
    
    Sequential(
      (0): create_conv_lin_nd_head(
        (0): Conv1d(128, 10, kernel_size=(1,), stride=(1,))
        (1): Linear(in_features=12, out_features=10, bias=True)
        (2): Transpose(-1, -2)
        (3): Reshape(bs, 5, 2, 10)
      )
    )
    
    
    bs = 16
    n_vars = 6
    seq_len = 12
    c_out = 2
    xb = torch.rand(bs, n_vars, seq_len)
    net = MultiInceptionTimePlus([1,2,3], c_out, seq_len)
    print(net.to(xb.device)(xb).shape)
    net.head __
    
    
    torch.Size([16, 2])
    
    
    Sequential(
      (0): Sequential(
        (0): GAP1d(
          (gap): AdaptiveAvgPool1d(output_size=1)
          (flatten): Reshape(bs)
        )
        (1): LinBnDrop(
          (0): Linear(in_features=384, out_features=2, bias=True)
        )
      )
    )
    
    
    bs = 8
    c_in = 7  # aka channels, features, variables, dimensions
    c_out = 2
    seq_len = 10
    xb2 = torch.randn(bs, c_in, seq_len)
    model1 = MultiInceptionTimePlus([2, 5], c_out, seq_len)
    model2 = MultiInceptionTimePlus([[0,2,5], [0,1,3,4,6]], c_out, seq_len)
    test_eq(model1.to(xb2.device)(xb2).shape, (bs, c_out))
    test_eq(model1.to(xb2.device)(xb2).shape, model2.to(xb2.device)(xb2).shape)__
    
    
    from tsai.data.external import *
    from tsai.data.core import *
    from tsai.data.preprocessing import *__
    
    
    X, y, splits = get_UCR_data('NATOPS', split_data=False)
    tfms  = [None, [TSCategorize()]]
    batch_tfms = TSStandardize()
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
    model = InceptionTimePlus(dls.vars, dls.c, dls.len)
    xb,yb=first(dls.train)
    test_eq(model.to(xb.device)(xb).shape, (dls.bs, dls.c))
    test_eq(count_parameters(model), 460038)__
    
    
    X, y, splits = get_UCR_data('NATOPS', split_data=False)
    tfms  = [None, [TSCategorize()]]
    batch_tfms = TSStandardize()
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
    model = MultiInceptionTimePlus([4, 15, 5], dls.c, dls.len)
    xb,yb=first(dls.train)
    test_eq(model.to(xb.device)(xb).shape, (dls.bs, dls.c))
    test_eq(count_parameters(model), 1370886)__

  * __Report an issue


