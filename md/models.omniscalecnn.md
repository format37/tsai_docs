## On this page

  * generate_layer_parameter_list
  * get_out_channel_number
  * get_Prime_number_in_a_range
  * OmniScaleCNN
  * build_layer_with_layer_parameter
  * SampaddingConv1D_BN



  * __Report an issue



  1. Models
  2. CNNs
  3. OmniScaleCNN



# OmniScaleCNN

> This is an unofficial PyTorch implementation created by Ignacio Oguiza - oguiza@timeseriesAI.co

* * *

source

### generate_layer_parameter_list

> 
>      generate_layer_parameter_list (start, end, layers, in_channel=1)

* * *

source

### get_out_channel_number

> 
>      get_out_channel_number (paramenter_layer, in_channel, prime_list)

* * *

source

### get_Prime_number_in_a_range

> 
>      get_Prime_number_in_a_range (start, end)

* * *

source

### OmniScaleCNN

> 
>      OmniScaleCNN (c_in, c_out, seq_len, layers=[1024, 229376],
>                    few_shot=False)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### build_layer_with_layer_parameter

> 
>      build_layer_with_layer_parameter (layer_parameters)

_formerly build_layer_with_layer_parameter_

* * *

source

### SampaddingConv1D_BN

> 
>      SampaddingConv1D_BN (in_channels, out_channels, kernel_size)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    bs = 16
    c_in = 3
    seq_len = 12
    c_out = 2
    xb = torch.rand(bs, c_in, seq_len)
    m = create_model(OmniScaleCNN, c_in, c_out, seq_len)
    test_eq(OmniScaleCNN(c_in, c_out, seq_len)(xb).shape, [bs, c_out])
    m __
    
    
    OmniScaleCNN(
      (net): Sequential(
        (0): build_layer_with_layer_parameter(
          (conv_list): ModuleList(
            (0): SampaddingConv1D_BN(
              (padding): ConstantPad1d(padding=(0, 0), value=0)
              (conv1d): Conv1d(3, 56, kernel_size=(1,), stride=(1,))
              (bn): BatchNorm1d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): SampaddingConv1D_BN(
              (padding): ConstantPad1d(padding=(0, 1), value=0)
              (conv1d): Conv1d(3, 56, kernel_size=(2,), stride=(1,))
              (bn): BatchNorm1d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (2): SampaddingConv1D_BN(
              (padding): ConstantPad1d(padding=(1, 1), value=0)
              (conv1d): Conv1d(3, 56, kernel_size=(3,), stride=(1,))
              (bn): BatchNorm1d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
        (1): build_layer_with_layer_parameter(
          (conv_list): ModuleList(
            (0): SampaddingConv1D_BN(
              (padding): ConstantPad1d(padding=(0, 0), value=0)
              (conv1d): Conv1d(168, 227, kernel_size=(1,), stride=(1,))
              (bn): BatchNorm1d(227, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): SampaddingConv1D_BN(
              (padding): ConstantPad1d(padding=(0, 1), value=0)
              (conv1d): Conv1d(168, 227, kernel_size=(2,), stride=(1,))
              (bn): BatchNorm1d(227, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (2): SampaddingConv1D_BN(
              (padding): ConstantPad1d(padding=(1, 1), value=0)
              (conv1d): Conv1d(168, 227, kernel_size=(3,), stride=(1,))
              (bn): BatchNorm1d(227, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
        (2): build_layer_with_layer_parameter(
          (conv_list): ModuleList(
            (0): SampaddingConv1D_BN(
              (padding): ConstantPad1d(padding=(0, 0), value=0)
              (conv1d): Conv1d(681, 510, kernel_size=(1,), stride=(1,))
              (bn): BatchNorm1d(510, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): SampaddingConv1D_BN(
              (padding): ConstantPad1d(padding=(0, 1), value=0)
              (conv1d): Conv1d(681, 510, kernel_size=(2,), stride=(1,))
              (bn): BatchNorm1d(510, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
      )
      (gap): GAP1d(
        (gap): AdaptiveAvgPool1d(output_size=1)
        (flatten): Flatten(full=False)
      )
      (hidden): Linear(in_features=1020, out_features=2, bias=True)
    )

  * __Report an issue


