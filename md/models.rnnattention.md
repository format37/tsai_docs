## On this page

  * Arguments
  * Imports
  * RNNAttention
    * GRUAttention
    * LSTMAttention
    * RNNAttention



  * __Report an issue



  1. Models
  2. RNNs
  3. RNNAttention



# RNNAttention

This is an custom PyTorch implementation by @yangtzech, based on TST implementation of Ignacio Oguiza.

## Arguments

Usual values are the ones that appear in the “Attention is all you need” and “A Transformer-based Framework for Multivariate Time Series Representation Learning” papers. And some parameters are necessary for the RNN part.

The default values are the ones selected as a default configuration in the latter.

  * c_in: the number of features (aka variables, dimensions, channels) in the time series dataset. dls.var
  * c_out: the number of target classes. dls.c
  * seq_len: number of time steps in the time series. dls.len
  * hidden_size: the number of features in the hidden state in the RNN model. Default: 128.
  * rnn_layers: the number of recurrent layers of the RNN model. Default: 1.
  * bias: If `False`, then the layer does not use bias weights `b_ih` and `b_hh`. Default: `True`
  * rnn_dropout: If non-zero, introduces a `Dropout` layer on the outputs of each RNN layer except the last layer, with dropout probability equal to :attr:`rnn_dropout`. Default: 0
  * bidirectional: If `True`, becomes a bidirectional RNN. Default: `False`
  * n_heads: parallel attention heads. Usual values: 8-16. Default: 16.
  * d_k: size of the learned linear projection of queries and keys in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
  * d_v: size of the learned linear projection of values in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
  * d_ff: the dimension of the feedforward network model. Usual values: 256-4096. Default: 256.
  * encoder_dropout: amount of residual dropout applied in the encoder. Usual values: 0.-0.3. Default: 0.1.
  * act: the activation function of intermediate layer, relu or gelu. Default: ‘gelu’.
  * encoder_layers: the number of sub-encoder-layers in the encoder. Usual values: 2-8. Default: 3.
  * fc_dropout: dropout applied to the final fully connected layer. Usual values: 0.-0.8. Default: 0.
  * y_range: range of possible y values (used in regression tasks). Default: None
  * kwargs: nn.Conv1d kwargs. If not {}, a nn.Conv1d with those kwargs will be applied to original time series.



## Imports

## RNNAttention
    
    
    t = torch.rand(16, 50, 128)
    output, attn = _MultiHeadAttention(d_model=128, n_heads=3, d_k=8, d_v=6)(t, t, t)
    output.shape, attn.shape __
    
    
    (torch.Size([16, 50, 128]), torch.Size([16, 3, 50, 50]))
    
    
    t = torch.rand(16, 50, 128)
    output = _TSTEncoderLayer(q_len=50, d_model=128, n_heads=3, d_k=None, d_v=None, d_ff=512, dropout=0.1, activation='gelu')(t)
    output.shape __
    
    
    torch.Size([16, 50, 128])

* * *

source

### GRUAttention

> 
>      GRUAttention (c_in:int, c_out:int, seq_len:int, hidden_size=128,
>                    rnn_layers=1, bias=True, rnn_dropout=0,
>                    bidirectional=False, encoder_layers:int=3, n_heads:int=16,
>                    d_k:Optional[int]=None, d_v:Optional[int]=None,
>                    d_ff:int=256, encoder_dropout:float=0.1, act:str='gelu',
>                    fc_dropout:float=0.0, y_range:Optional[tuple]=None,
>                    verbose:bool=False, custom_head=None)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### LSTMAttention

> 
>      LSTMAttention (c_in:int, c_out:int, seq_len:int, hidden_size=128,
>                     rnn_layers=1, bias=True, rnn_dropout=0,
>                     bidirectional=False, encoder_layers:int=3, n_heads:int=16,
>                     d_k:Optional[int]=None, d_v:Optional[int]=None,
>                     d_ff:int=256, encoder_dropout:float=0.1, act:str='gelu',
>                     fc_dropout:float=0.0, y_range:Optional[tuple]=None,
>                     verbose:bool=False, custom_head=None)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### RNNAttention

> 
>      RNNAttention (c_in:int, c_out:int, seq_len:int, hidden_size=128,
>                    rnn_layers=1, bias=True, rnn_dropout=0,
>                    bidirectional=False, encoder_layers:int=3, n_heads:int=16,
>                    d_k:Optional[int]=None, d_v:Optional[int]=None,
>                    d_ff:int=256, encoder_dropout:float=0.1, act:str='gelu',
>                    fc_dropout:float=0.0, y_range:Optional[tuple]=None,
>                    verbose:bool=False, custom_head=None)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    bs = 32
    c_in = 9  # aka channels, features, variables, dimensions
    c_out = 2
    seq_len = 500
    
    xb = torch.randn(bs, c_in, seq_len)
    
    # standardize by channel by_var based on the training set
    xb = (xb - xb.mean((0, 2), keepdim=True)) / xb.std((0, 2), keepdim=True)
    
    # Settings
    hidden_size = 128
    rnn_layers=1
    bias=True
    rnn_dropout=0
    bidirectional=False
    encoder_layers=3
    n_heads = 16
    d_k = d_v = None # if None --> d_model // n_heads
    d_ff = 256
    encoder_dropout = 0.1
    act = "gelu"
    fc_dropout = 0.1
    kwargs = {}
    
    model = RNNAttention(c_in, c_out, seq_len, hidden_size=hidden_size, rnn_layers=rnn_layers, bias=bias, rnn_dropout=rnn_dropout, bidirectional=bidirectional,
                encoder_layers=encoder_layers, n_heads=n_heads,
                d_k=d_k, d_v=d_v, d_ff=d_ff, encoder_dropout=encoder_dropout, act=act, 
                fc_dropout=fc_dropout, **kwargs)
    test_eq(model.to(xb.device)(xb).shape, [bs, c_out])
    print(f'model parameters: {count_parameters(model)}')__
    
    
    model parameters: 541698
    
    
    bs = 32
    c_in = 9  # aka channels, features, variables, dimensions
    c_out = 2
    seq_len = 60
    
    xb = torch.randn(bs, c_in, seq_len)
    
    # standardize by channel by_var based on the training set
    xb = (xb - xb.mean((0, 2), keepdim=True)) / xb.std((0, 2), keepdim=True)
    
    # Settings
    hidden_size = 128
    rnn_layers=1
    bias=True
    rnn_dropout=0
    bidirectional=False
    encoder_layers=3
    n_heads = 16
    d_k = d_v = None # if None --> d_model // n_heads
    d_ff = 256
    encoder_dropout = 0.1
    act = "gelu"
    fc_dropout = 0.1
    kwargs = {}
    # kwargs = dict(kernel_size=5, padding=2)
    
    model = RNNAttention(c_in, c_out, seq_len, hidden_size=hidden_size, rnn_layers=rnn_layers, bias=bias, rnn_dropout=rnn_dropout, bidirectional=bidirectional,
                encoder_layers=encoder_layers, n_heads=n_heads,
                d_k=d_k, d_v=d_v, d_ff=d_ff, encoder_dropout=encoder_dropout, act=act, 
                fc_dropout=fc_dropout, **kwargs)
    test_eq(model.to(xb.device)(xb).shape, [bs, c_out])
    print(f'model parameters: {count_parameters(model)}')__
    
    
    model parameters: 429058

  * __Report an issue


