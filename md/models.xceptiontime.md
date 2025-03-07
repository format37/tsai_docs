## On this page

  * XceptionTime
  * XceptionBlock
  * XceptionModule



  * __Report an issue



  1. Models
  2. CNNs
  3. XceptionTime



# XceptionTime

This is an unofficial PyTorch implementation by Ignacio Oguiza - oguiza@timeseriesAI.co modified on:

Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J. & Petitjean, F. (2019). **InceptionTime: Finding AlexNet for Time Series Classification**. arXiv preprint arXiv:1909.04939.

Official InceptionTime tensorflow implementation: https://github.com/hfawaz/InceptionTime

* * *

source

### XceptionTime

> 
>      XceptionTime (c_in, c_out, nf=16, nb_filters=None, adaptive_size=50,
>                    residual=True)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### XceptionBlock

> 
>      XceptionBlock (ni, nf, residual=True, ks=40, bottleneck=True)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### XceptionModule

> 
>      XceptionModule (ni, nf, ks=40, bottleneck=True)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    bs = 16
    vars = 3
    seq_len = 12
    c_out = 6
    xb = torch.rand(bs, vars, seq_len)
    test_eq(XceptionTime(vars,c_out)(xb).shape, [bs, c_out])
    test_eq(XceptionTime(vars,c_out, bottleneck=False)(xb).shape, [bs, c_out])
    test_eq(XceptionTime(vars,c_out, residual=False)(xb).shape, [bs, c_out])
    test_eq(count_parameters(XceptionTime(3, 2)), 399540)__
    
    
    m = XceptionTime(2,3)
    test_eq(check_weight(m, is_bn)[0].sum(), 5) # 2 shortcut + 3 bn
    test_eq(len(check_bias(m, is_conv)[0]), 0)
    test_eq(len(check_bias(m)[0]), 5) # 2 shortcut + 3 bn __
    
    
    XceptionTime(3, 2)__
    
    
    XceptionTime(
      (block): XceptionBlock(
        (xception): ModuleList(
          (0): XceptionModule(
            (bottleneck): Conv1d(3, 16, kernel_size=(1,), stride=(1,), bias=False)
            (convs): ModuleList(
              (0): SeparableConv1d(
                (depthwise_conv): Conv1d(16, 16, kernel_size=(39,), stride=(1,), padding=(19,), groups=16, bias=False)
                (pointwise_conv): Conv1d(16, 16, kernel_size=(1,), stride=(1,), bias=False)
              )
              (1): SeparableConv1d(
                (depthwise_conv): Conv1d(16, 16, kernel_size=(19,), stride=(1,), padding=(9,), groups=16, bias=False)
                (pointwise_conv): Conv1d(16, 16, kernel_size=(1,), stride=(1,), bias=False)
              )
              (2): SeparableConv1d(
                (depthwise_conv): Conv1d(16, 16, kernel_size=(9,), stride=(1,), padding=(4,), groups=16, bias=False)
                (pointwise_conv): Conv1d(16, 16, kernel_size=(1,), stride=(1,), bias=False)
              )
            )
            (maxconvpool): Sequential(
              (0): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
              (1): Conv1d(3, 16, kernel_size=(1,), stride=(1,), bias=False)
            )
            (concat): Concat(dim=1)
          )
          (1): XceptionModule(
            (bottleneck): Conv1d(64, 32, kernel_size=(1,), stride=(1,), bias=False)
            (convs): ModuleList(
              (0): SeparableConv1d(
                (depthwise_conv): Conv1d(32, 32, kernel_size=(39,), stride=(1,), padding=(19,), groups=32, bias=False)
                (pointwise_conv): Conv1d(32, 32, kernel_size=(1,), stride=(1,), bias=False)
              )
              (1): SeparableConv1d(
                (depthwise_conv): Conv1d(32, 32, kernel_size=(19,), stride=(1,), padding=(9,), groups=32, bias=False)
                (pointwise_conv): Conv1d(32, 32, kernel_size=(1,), stride=(1,), bias=False)
              )
              (2): SeparableConv1d(
                (depthwise_conv): Conv1d(32, 32, kernel_size=(9,), stride=(1,), padding=(4,), groups=32, bias=False)
                (pointwise_conv): Conv1d(32, 32, kernel_size=(1,), stride=(1,), bias=False)
              )
            )
            (maxconvpool): Sequential(
              (0): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
              (1): Conv1d(64, 32, kernel_size=(1,), stride=(1,), bias=False)
            )
            (concat): Concat(dim=1)
          )
          (2): XceptionModule(
            (bottleneck): Conv1d(128, 64, kernel_size=(1,), stride=(1,), bias=False)
            (convs): ModuleList(
              (0): SeparableConv1d(
                (depthwise_conv): Conv1d(64, 64, kernel_size=(39,), stride=(1,), padding=(19,), groups=64, bias=False)
                (pointwise_conv): Conv1d(64, 64, kernel_size=(1,), stride=(1,), bias=False)
              )
              (1): SeparableConv1d(
                (depthwise_conv): Conv1d(64, 64, kernel_size=(19,), stride=(1,), padding=(9,), groups=64, bias=False)
                (pointwise_conv): Conv1d(64, 64, kernel_size=(1,), stride=(1,), bias=False)
              )
              (2): SeparableConv1d(
                (depthwise_conv): Conv1d(64, 64, kernel_size=(9,), stride=(1,), padding=(4,), groups=64, bias=False)
                (pointwise_conv): Conv1d(64, 64, kernel_size=(1,), stride=(1,), bias=False)
              )
            )
            (maxconvpool): Sequential(
              (0): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
              (1): Conv1d(128, 64, kernel_size=(1,), stride=(1,), bias=False)
            )
            (concat): Concat(dim=1)
          )
          (3): XceptionModule(
            (bottleneck): Conv1d(256, 128, kernel_size=(1,), stride=(1,), bias=False)
            (convs): ModuleList(
              (0): SeparableConv1d(
                (depthwise_conv): Conv1d(128, 128, kernel_size=(39,), stride=(1,), padding=(19,), groups=128, bias=False)
                (pointwise_conv): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
              )
              (1): SeparableConv1d(
                (depthwise_conv): Conv1d(128, 128, kernel_size=(19,), stride=(1,), padding=(9,), groups=128, bias=False)
                (pointwise_conv): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
              )
              (2): SeparableConv1d(
                (depthwise_conv): Conv1d(128, 128, kernel_size=(9,), stride=(1,), padding=(4,), groups=128, bias=False)
                (pointwise_conv): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
              )
            )
            (maxconvpool): Sequential(
              (0): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
              (1): Conv1d(256, 128, kernel_size=(1,), stride=(1,), bias=False)
            )
            (concat): Concat(dim=1)
          )
        )
        (shortcut): ModuleList(
          (0): ConvBlock(
            (0): Conv1d(3, 128, kernel_size=(1,), stride=(1,), bias=False)
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (0): Conv1d(128, 512, kernel_size=(1,), stride=(1,), bias=False)
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (add): Add
        (act): ReLU()
      )
      (head): Sequential(
        (0): AdaptiveAvgPool1d(output_size=50)
        (1): ConvBlock(
          (0): Conv1d(512, 256, kernel_size=(1,), stride=(1,), bias=False)
          (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (2): ConvBlock(
          (0): Conv1d(256, 128, kernel_size=(1,), stride=(1,), bias=False)
          (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (3): ConvBlock(
          (0): Conv1d(128, 2, kernel_size=(1,), stride=(1,), bias=False)
          (1): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (4): GAP1d(
          (gap): AdaptiveAvgPool1d(output_size=1)
          (flatten): Flatten(full=False)
        )
      )
    )

  * __Report an issue


