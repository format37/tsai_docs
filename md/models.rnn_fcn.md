## On this page

  * MGRU_FCN
  * MLSTM_FCN
  * MRNN_FCN
  * GRU_FCN
  * LSTM_FCN
  * RNN_FCN



  * __Report an issue



  1. Models
  2. Hybrid models
  3. RNN_FCN



# RNN_FCN

> This is an unofficial PyTorch implementation created by Ignacio Oguiza - oguiza@timeseriesAI.co

* * *

source

### MGRU_FCN

> 
>      MGRU_FCN (*args, se=16, **kwargs)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### MLSTM_FCN

> 
>      MLSTM_FCN (*args, se=16, **kwargs)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### MRNN_FCN

> 
>      MRNN_FCN (*args, se=16, **kwargs)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### GRU_FCN

> 
>      GRU_FCN (c_in, c_out, seq_len=None, hidden_size=100, rnn_layers=1,
>               bias=True, cell_dropout=0, rnn_dropout=0.8, bidirectional=False,
>               shuffle=True, fc_dropout=0.0, conv_layers=[128, 256, 128],
>               kss=[7, 5, 3], se=0)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### LSTM_FCN

> 
>      LSTM_FCN (c_in, c_out, seq_len=None, hidden_size=100, rnn_layers=1,
>                bias=True, cell_dropout=0, rnn_dropout=0.8,
>                bidirectional=False, shuffle=True, fc_dropout=0.0,
>                conv_layers=[128, 256, 128], kss=[7, 5, 3], se=0)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### RNN_FCN

> 
>      RNN_FCN (c_in, c_out, seq_len=None, hidden_size=100, rnn_layers=1,
>               bias=True, cell_dropout=0, rnn_dropout=0.8, bidirectional=False,
>               shuffle=True, fc_dropout=0.0, conv_layers=[128, 256, 128],
>               kss=[7, 5, 3], se=0)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    bs = 16
    n_vars = 3
    seq_len = 12
    c_out = 2
    xb = torch.rand(bs, n_vars, seq_len)
    test_eq(RNN_FCN(n_vars, c_out, seq_len)(xb).shape, [bs, c_out])
    test_eq(LSTM_FCN(n_vars, c_out, seq_len)(xb).shape, [bs, c_out])
    test_eq(MLSTM_FCN(n_vars, c_out, seq_len)(xb).shape, [bs, c_out])
    test_eq(GRU_FCN(n_vars, c_out, shuffle=False)(xb).shape, [bs, c_out])
    test_eq(GRU_FCN(n_vars, c_out, seq_len, shuffle=False)(xb).shape, [bs, c_out])__
    
    
    LSTM_FCN(n_vars, seq_len, c_out, se=8)__
    
    
    LSTM_FCN(
      (rnn): LSTM(2, 100, batch_first=True)
      (rnn_dropout): Dropout(p=0.8, inplace=False)
      (convblock1): ConvBlock(
        (0): Conv1d(3, 128, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (se1): SqueezeExciteBlock(
        (avg_pool): GAP1d(
          (gap): AdaptiveAvgPool1d(output_size=1)
          (flatten): Flatten(full=False)
        )
        (fc): Sequential(
          (0): Linear(in_features=128, out_features=16, bias=False)
          (1): ReLU()
          (2): Linear(in_features=16, out_features=128, bias=False)
          (3): Sigmoid()
        )
      )
      (convblock2): ConvBlock(
        (0): Conv1d(128, 256, kernel_size=(5,), stride=(1,), padding=(2,), bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (se2): SqueezeExciteBlock(
        (avg_pool): GAP1d(
          (gap): AdaptiveAvgPool1d(output_size=1)
          (flatten): Flatten(full=False)
        )
        (fc): Sequential(
          (0): Linear(in_features=256, out_features=32, bias=False)
          (1): ReLU()
          (2): Linear(in_features=32, out_features=256, bias=False)
          (3): Sigmoid()
        )
      )
      (convblock3): ConvBlock(
        (0): Conv1d(256, 128, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (gap): GAP1d(
        (gap): AdaptiveAvgPool1d(output_size=1)
        (flatten): Flatten(full=False)
      )
      (concat): Concat(dim=1)
      (fc): Linear(in_features=228, out_features=12, bias=True)
    )

  * __Report an issue


