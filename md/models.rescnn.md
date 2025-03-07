## On this page

  * ResCNN



  * __Report an issue



  1. Models
  2. CNNs
  3. ResCNN



# ResCNN

> This is an unofficial PyTorch implementation by Ignacio Oguiza - oguiza@timeseriesAI.co
    
    
    from tsai.models.utils import *__

* * *

source

### ResCNN

> 
>      ResCNN (c_in, c_out, coord=False, separable=False, zero_norm=False)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    xb = torch.rand(16, 3, 10)
    test_eq(ResCNN(3,2,coord=True, separable=True)(xb).shape, [xb.shape[0], 2])
    test_eq(count_parameters(ResCNN(3,2)), 257283)__
    
    
    ResCNN(3,2,coord=True, separable=True)__
    
    
    ResCNN(
      (block1): _ResCNNBlock(
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
      (block2): ConvBlock(
        (0): AddCoords1d()
        (1): SeparableConv1d(
          (depthwise_conv): Conv1d(65, 65, kernel_size=(3,), stride=(1,), padding=(1,), groups=65, bias=False)
          (pointwise_conv): Conv1d(65, 128, kernel_size=(1,), stride=(1,), bias=False)
        )
        (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): LeakyReLU(negative_slope=0.2)
      )
      (block3): ConvBlock(
        (0): AddCoords1d()
        (1): SeparableConv1d(
          (depthwise_conv): Conv1d(129, 129, kernel_size=(3,), stride=(1,), padding=(1,), groups=129, bias=False)
          (pointwise_conv): Conv1d(129, 256, kernel_size=(1,), stride=(1,), bias=False)
        )
        (2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): PReLU(num_parameters=1)
      )
      (block4): ConvBlock(
        (0): AddCoords1d()
        (1): SeparableConv1d(
          (depthwise_conv): Conv1d(257, 257, kernel_size=(3,), stride=(1,), padding=(1,), groups=257, bias=False)
          (pointwise_conv): Conv1d(257, 128, kernel_size=(1,), stride=(1,), bias=False)
        )
        (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ELU(alpha=0.3)
      )
      (gap): AdaptiveAvgPool1d(output_size=1)
      (squeeze): Squeeze(dim=-1)
      (lin): Linear(in_features=128, out_features=2, bias=True)
    )
    
    
    check_weight(ResCNN(3,2, zero_norm=True), is_bn)__
    
    
    (array([1., 1., 0., 1., 1., 1., 1.], dtype=float32),
     array([0., 0., 0., 0., 0., 0., 0.], dtype=float32))

  * __Report an issue


