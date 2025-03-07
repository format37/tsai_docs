## On this page

  * ResNet
  * ResBlock



  * __Report an issue



  1. Models
  2. CNNs
  3. ResNet



# ResNet

> This is an unofficial PyTorch implementation created by Ignacio Oguiza - oguiza@timeseriesAI.co

* * *

source

### ResNet

> 
>      ResNet (c_in, c_out)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### ResBlock

> 
>      ResBlock (ni, nf, kss=[7, 5, 3])

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    xb = torch.rand(2, 3, 4)
    test_eq(ResNet(3,2)(xb).shape, [xb.shape[0], 2])
    test_eq(count_parameters(ResNet(3, 2)), 479490) # for (3,2)__
    
    
    ResNet(3,2)__
    
    
    ResNet(
      (resblock1): ResBlock(
        (convblock1): ConvBlock(
          (0): Conv1d(3, 64, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
          (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (convblock2): ConvBlock(
          (0): Conv1d(64, 64, kernel_size=(5,), stride=(1,), padding=(2,), bias=False)
          (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (convblock3): ConvBlock(
          (0): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
          (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (shortcut): ConvBlock(
          (0): Conv1d(3, 64, kernel_size=(1,), stride=(1,), bias=False)
          (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (add): Add
        (act): ReLU()
      )
      (resblock2): ResBlock(
        (convblock1): ConvBlock(
          (0): Conv1d(64, 128, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
          (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (convblock2): ConvBlock(
          (0): Conv1d(128, 128, kernel_size=(5,), stride=(1,), padding=(2,), bias=False)
          (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (convblock3): ConvBlock(
          (0): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
          (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (shortcut): ConvBlock(
          (0): Conv1d(64, 128, kernel_size=(1,), stride=(1,), bias=False)
          (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (add): Add
        (act): ReLU()
      )
      (resblock3): ResBlock(
        (convblock1): ConvBlock(
          (0): Conv1d(128, 128, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
          (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (convblock2): ConvBlock(
          (0): Conv1d(128, 128, kernel_size=(5,), stride=(1,), padding=(2,), bias=False)
          (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (convblock3): ConvBlock(
          (0): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
          (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (shortcut): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (add): Add
        (act): ReLU()
      )
      (gap): AdaptiveAvgPool1d(output_size=1)
      (squeeze): Squeeze(dim=-1)
      (fc): Linear(in_features=128, out_features=2, bias=True)
    )

  * __Report an issue


