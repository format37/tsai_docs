## On this page

  * Arguments
  * Imports
  * RNNAttentionPlus
    * GRUAttentionPlus
    * LSTMAttentionPlus
    * RNNAttentionPlus



  * __Report an issue



# RNNAttentionPlus

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



## Imports

## RNNAttentionPlus
    
    
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

### GRUAttentionPlus

> 
>      GRUAttentionPlus (c_in:int, c_out:int, seq_len:int, d:tuple=None,
>                        hidden_size:int=128, rnn_layers:int=1, bias:bool=True,
>                        rnn_dropout:float=0, bidirectional=False,
>                        encoder_layers:int=3, n_heads:int=16,
>                        d_k:Optional[int]=None, d_v:Optional[int]=None,
>                        d_ff:int=256, encoder_dropout:float=0.1,
>                        act:str='gelu', fc_dropout:float=0.0,
>                        y_range:Optional[tuple]=None, custom_head=None,
>                        use_bn:bool=True, flatten:bool=True)

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

| **Type** | **Default** | **Details**  
---|---|---|---  
c_in | int |  | the number of features (aka variables, dimensions, channels) in the time series dataset.  
c_out | int |  | the number of target classes.  
seq_len | int |  | number of time steps in the time series.  
d | tuple | None | output shape (excluding batch dimension).  
hidden_size | int | 128 | the number of features in the hidden state h  
rnn_layers | int | 1 | the number of recurrent layers of the RNN model.  
bias | bool | True | If `False`, then the layer does not use bias weights `b_ih` and `b_hh`.  
rnn_dropout | float | 0 | rnn dropout applied to the output of each RNN layer except the last layer.  
bidirectional | bool | False | If `True`, becomes a bidirectional RNN. Default: `False`  
encoder_layers | int | 3 | the number of sub-encoder-layers in the encoder.  
n_heads | int | 16 | parallel attention heads.  
d_k | Optional | None | size of the learned linear projection of queries and keys in the MHA.  
d_v | Optional | None | size of the learned linear projection of values in the MHA.  
d_ff | int | 256 | the dimension of the feedforward network model.  
encoder_dropout | float | 0.1 | amount of residual dropout applied in the encoder.  
act | str | gelu | the activation function of intermediate layer, relu or gelu.  
fc_dropout | float | 0.0 | dropout applied to the final fully connected layer.  
y_range | Optional | None | range of possible y values (used in regression tasks).  
custom_head | NoneType | None | custom head that will be applied to the model head (optional).  
use_bn | bool | True | indicates if batchnorm will be applied to the model head.  
flatten | bool | True | this will flatten the output of the encoder before applying the head if True.  
  
* * *

source

### LSTMAttentionPlus

> 
>      LSTMAttentionPlus (c_in:int, c_out:int, seq_len:int, d:tuple=None,
>                         hidden_size:int=128, rnn_layers:int=1, bias:bool=True,
>                         rnn_dropout:float=0, bidirectional=False,
>                         encoder_layers:int=3, n_heads:int=16,
>                         d_k:Optional[int]=None, d_v:Optional[int]=None,
>                         d_ff:int=256, encoder_dropout:float=0.1,
>                         act:str='gelu', fc_dropout:float=0.0,
>                         y_range:Optional[tuple]=None, custom_head=None,
>                         use_bn:bool=True, flatten:bool=True)

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

| **Type** | **Default** | **Details**  
---|---|---|---  
c_in | int |  | the number of features (aka variables, dimensions, channels) in the time series dataset.  
c_out | int |  | the number of target classes.  
seq_len | int |  | number of time steps in the time series.  
d | tuple | None | output shape (excluding batch dimension).  
hidden_size | int | 128 | the number of features in the hidden state h  
rnn_layers | int | 1 | the number of recurrent layers of the RNN model.  
bias | bool | True | If `False`, then the layer does not use bias weights `b_ih` and `b_hh`.  
rnn_dropout | float | 0 | rnn dropout applied to the output of each RNN layer except the last layer.  
bidirectional | bool | False | If `True`, becomes a bidirectional RNN. Default: `False`  
encoder_layers | int | 3 | the number of sub-encoder-layers in the encoder.  
n_heads | int | 16 | parallel attention heads.  
d_k | Optional | None | size of the learned linear projection of queries and keys in the MHA.  
d_v | Optional | None | size of the learned linear projection of values in the MHA.  
d_ff | int | 256 | the dimension of the feedforward network model.  
encoder_dropout | float | 0.1 | amount of residual dropout applied in the encoder.  
act | str | gelu | the activation function of intermediate layer, relu or gelu.  
fc_dropout | float | 0.0 | dropout applied to the final fully connected layer.  
y_range | Optional | None | range of possible y values (used in regression tasks).  
custom_head | NoneType | None | custom head that will be applied to the model head (optional).  
use_bn | bool | True | indicates if batchnorm will be applied to the model head.  
flatten | bool | True | this will flatten the output of the encoder before applying the head if True.  
  
* * *

source

### RNNAttentionPlus

> 
>      RNNAttentionPlus (c_in:int, c_out:int, seq_len:int, d:tuple=None,
>                        hidden_size:int=128, rnn_layers:int=1, bias:bool=True,
>                        rnn_dropout:float=0, bidirectional=False,
>                        encoder_layers:int=3, n_heads:int=16,
>                        d_k:Optional[int]=None, d_v:Optional[int]=None,
>                        d_ff:int=256, encoder_dropout:float=0.1,
>                        act:str='gelu', fc_dropout:float=0.0,
>                        y_range:Optional[tuple]=None, custom_head=None,
>                        use_bn:bool=True, flatten:bool=True)

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

| **Type** | **Default** | **Details**  
---|---|---|---  
c_in | int |  | the number of features (aka variables, dimensions, channels) in the time series dataset.  
c_out | int |  | the number of target classes.  
seq_len | int |  | number of time steps in the time series.  
d | tuple | None | output shape (excluding batch dimension).  
hidden_size | int | 128 | the number of features in the hidden state h  
rnn_layers | int | 1 | the number of recurrent layers of the RNN model.  
bias | bool | True | If `False`, then the layer does not use bias weights `b_ih` and `b_hh`.  
rnn_dropout | float | 0 | rnn dropout applied to the output of each RNN layer except the last layer.  
bidirectional | bool | False | If `True`, becomes a bidirectional RNN. Default: `False`  
encoder_layers | int | 3 | the number of sub-encoder-layers in the encoder.  
n_heads | int | 16 | parallel attention heads.  
d_k | Optional | None | size of the learned linear projection of queries and keys in the MHA.  
d_v | Optional | None | size of the learned linear projection of values in the MHA.  
d_ff | int | 256 | the dimension of the feedforward network model.  
encoder_dropout | float | 0.1 | amount of residual dropout applied in the encoder.  
act | str | gelu | the activation function of intermediate layer, relu or gelu.  
fc_dropout | float | 0.0 | dropout applied to the final fully connected layer.  
y_range | Optional | None | range of possible y values (used in regression tasks).  
custom_head | NoneType | None | custom head that will be applied to the model head (optional).  
use_bn | bool | True | indicates if batchnorm will be applied to the model head.  
flatten | bool | True | this will flatten the output of the encoder before applying the head if True.  
      
    
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
    
    model = RNNAttentionPlus(c_in, c_out, seq_len, hidden_size=hidden_size, rnn_layers=rnn_layers, bias=bias, rnn_dropout=rnn_dropout, bidirectional=bidirectional,
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
    
    model = RNNAttentionPlus(c_in, c_out, seq_len, hidden_size=hidden_size, rnn_layers=rnn_layers, bias=bias, rnn_dropout=rnn_dropout, bidirectional=bidirectional,
                encoder_layers=encoder_layers, n_heads=n_heads,
                d_k=d_k, d_v=d_v, d_ff=d_ff, encoder_dropout=encoder_dropout, act=act, 
                fc_dropout=fc_dropout, **kwargs)
    test_eq(model.to(xb.device)(xb).shape, [bs, c_out])
    print(f'model parameters: {count_parameters(model)}')__
    
    
    model parameters: 429058
    
    
    bs = 32
    c_in = 9  # aka channels, features, variables, dimensions
    c_out = 2
    seq_len = 60
    d = 10
    
    xb = torch.randn(bs, c_in, seq_len)
    model = RNNAttentionPlus(c_in, c_out, seq_len, d=d)
    test_eq(model.to(xb.device)(xb).shape, [bs, d, c_out])
    print(f'model parameters: {count_parameters(model)}')__
    
    
    model parameters: 567572
    
    
    bs = 32
    c_in = 9  # aka channels, features, variables, dimensions
    c_out = 2
    seq_len = 60
    d = (3, 10)
    
    xb = torch.randn(bs, c_in, seq_len)
    model = RNNAttentionPlus(c_in, c_out, seq_len, d=d)
    test_eq(model.to(xb.device)(xb).shape, [bs, *d, c_out])
    print(f'model parameters: {count_parameters(model)}')__
    
    
    model parameters: 874812

  * __Report an issue


