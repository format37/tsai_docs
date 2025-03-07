## On this page

  * InceptionTime
  * InceptionBlock
  * InceptionModule



  * __Report an issue



  1. Models
  2. CNNs
  3. InceptionTime



# InceptionTime

> An ensemble of deep Convolutional Neural Network (CNN) models, inspired by the Inception-v4 architecture

This is an unofficial PyTorch implementation created by Ignacio Oguiza (oguiza@timeseriesAI.co) based on:

Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J. & Petitjean, F. (2019). **InceptionTime: Finding AlexNet for Time Series Classification**. arXiv preprint arXiv:1909.04939.

Official InceptionTime tensorflow implementation: https://github.com/hfawaz/InceptionTime

* * *

source

### InceptionTime

> 
>      InceptionTime (c_in, c_out, seq_len=None, nf=32, nb_filters=None, ks=40,
>                     bottleneck=True)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### InceptionBlock

> 
>      InceptionBlock (ni, nf=32, residual=True, depth=6, ks=40,
>                      bottleneck=True)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### InceptionModule

> 
>      InceptionModule (ni, nf, ks=40, bottleneck=True)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    from tsai.models.utils import count_parameters __
    
    
    bs = 16
    vars = 1
    seq_len = 12
    c_out = 2
    xb = torch.rand(bs, vars, seq_len)
    test_eq(InceptionTime(vars,c_out)(xb).shape, [bs, c_out])
    test_eq(InceptionTime(vars,c_out, bottleneck=False)(xb).shape, [bs, c_out])
    test_eq(InceptionTime(vars,c_out, residual=False)(xb).shape, [bs, c_out])
    test_eq(count_parameters(InceptionTime(3, 2)), 455490)__
    
    
    InceptionTime(3,2)__
    
    
    InceptionTime(
      (inceptionblock): InceptionBlock(
        (inception): ModuleList(
          (0): InceptionModule(
            (bottleneck): Conv1d(3, 32, kernel_size=(1,), stride=(1,), bias=False)
            (convs): ModuleList(
              (0): Conv1d(32, 32, kernel_size=(39,), stride=(1,), padding=(19,), bias=False)
              (1): Conv1d(32, 32, kernel_size=(19,), stride=(1,), padding=(9,), bias=False)
              (2): Conv1d(32, 32, kernel_size=(9,), stride=(1,), padding=(4,), bias=False)
            )
            (maxconvpool): Sequential(
              (0): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
              (1): Conv1d(3, 32, kernel_size=(1,), stride=(1,), bias=False)
            )
            (concat): Concat(dim=1)
            (bn): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): ReLU()
          )
          (1): InceptionModule(
            (bottleneck): Conv1d(128, 32, kernel_size=(1,), stride=(1,), bias=False)
            (convs): ModuleList(
              (0): Conv1d(32, 32, kernel_size=(39,), stride=(1,), padding=(19,), bias=False)
              (1): Conv1d(32, 32, kernel_size=(19,), stride=(1,), padding=(9,), bias=False)
              (2): Conv1d(32, 32, kernel_size=(9,), stride=(1,), padding=(4,), bias=False)
            )
            (maxconvpool): Sequential(
              (0): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
              (1): Conv1d(128, 32, kernel_size=(1,), stride=(1,), bias=False)
            )
            (concat): Concat(dim=1)
            (bn): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): ReLU()
          )
          (2): InceptionModule(
            (bottleneck): Conv1d(128, 32, kernel_size=(1,), stride=(1,), bias=False)
            (convs): ModuleList(
              (0): Conv1d(32, 32, kernel_size=(39,), stride=(1,), padding=(19,), bias=False)
              (1): Conv1d(32, 32, kernel_size=(19,), stride=(1,), padding=(9,), bias=False)
              (2): Conv1d(32, 32, kernel_size=(9,), stride=(1,), padding=(4,), bias=False)
            )
            (maxconvpool): Sequential(
              (0): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
              (1): Conv1d(128, 32, kernel_size=(1,), stride=(1,), bias=False)
            )
            (concat): Concat(dim=1)
            (bn): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): ReLU()
          )
          (3): InceptionModule(
            (bottleneck): Conv1d(128, 32, kernel_size=(1,), stride=(1,), bias=False)
            (convs): ModuleList(
              (0): Conv1d(32, 32, kernel_size=(39,), stride=(1,), padding=(19,), bias=False)
              (1): Conv1d(32, 32, kernel_size=(19,), stride=(1,), padding=(9,), bias=False)
              (2): Conv1d(32, 32, kernel_size=(9,), stride=(1,), padding=(4,), bias=False)
            )
            (maxconvpool): Sequential(
              (0): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
              (1): Conv1d(128, 32, kernel_size=(1,), stride=(1,), bias=False)
            )
            (concat): Concat(dim=1)
            (bn): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): ReLU()
          )
          (4): InceptionModule(
            (bottleneck): Conv1d(128, 32, kernel_size=(1,), stride=(1,), bias=False)
            (convs): ModuleList(
              (0): Conv1d(32, 32, kernel_size=(39,), stride=(1,), padding=(19,), bias=False)
              (1): Conv1d(32, 32, kernel_size=(19,), stride=(1,), padding=(9,), bias=False)
              (2): Conv1d(32, 32, kernel_size=(9,), stride=(1,), padding=(4,), bias=False)
            )
            (maxconvpool): Sequential(
              (0): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
              (1): Conv1d(128, 32, kernel_size=(1,), stride=(1,), bias=False)
            )
            (concat): Concat(dim=1)
            (bn): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): ReLU()
          )
          (5): InceptionModule(
            (bottleneck): Conv1d(128, 32, kernel_size=(1,), stride=(1,), bias=False)
            (convs): ModuleList(
              (0): Conv1d(32, 32, kernel_size=(39,), stride=(1,), padding=(19,), bias=False)
              (1): Conv1d(32, 32, kernel_size=(19,), stride=(1,), padding=(9,), bias=False)
              (2): Conv1d(32, 32, kernel_size=(9,), stride=(1,), padding=(4,), bias=False)
            )
            (maxconvpool): Sequential(
              (0): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
              (1): Conv1d(128, 32, kernel_size=(1,), stride=(1,), bias=False)
            )
            (concat): Concat(dim=1)
            (bn): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): ReLU()
          )
        )
        (shortcut): ModuleList(
          (0): ConvBlock(
            (0): Conv1d(3, 128, kernel_size=(1,), stride=(1,), bias=False)
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (add): Add
        (act): ReLU()
      )
      (gap): GAP1d(
        (gap): AdaptiveAvgPool1d(output_size=1)
        (flatten): Flatten(full=False)
      )
      (fc): Linear(in_features=128, out_features=2, bias=True)
    )

  * __Report an issue


