## On this page

  * TransformerGRUPlus
  * TransformerLSTMPlus
  * TransformerRNNPlus



  * __Report an issue



  1. Models
  2. Hybrid models
  3. TransformerRNNPlus



# TransformerRNNPlus

These is a Pytorch implementation of a Transformer + RNN created by Ignacio Oguiza - oguiza@timeseriesAI.co inspired by the code created by Baurzhan Urazalinov (https://www.kaggle.com/baurzhanurazalinov).

Baurzhan Urazalinov won a Kaggle competition (Parkinson’s Freezing of Gait Prediction: Event detection from wearable sensor data - 2023) using the following original tensorflow code:

  * https://www.kaggle.com/code/baurzhanurazalinov/parkinson-s-freezing-defog-training-code
  * https://www.kaggle.com/code/baurzhanurazalinov/parkinson-s-freezing-tdcsfog-training-code
  * https://www.kaggle.com/code/baurzhanurazalinov/parkinson-s-freezing-submission-code



I’d like to congratulate Baurzhan for winning this competition, and for sharing the code he used.
    
    
    from tsai.models.utils import count_parameters __
    
    
    t = torch.rand(4, 864, 54)
    encoder_layer = torch.nn.TransformerEncoderLayer(54, 6, dim_feedforward=2048, dropout=0.1, 
                                                     activation="relu", layer_norm_eps=1e-05, 
                                                     batch_first=True, norm_first=False)
    print(encoder_layer(t).shape)
    print(count_parameters(encoder_layer))__
    
    
    torch.Size([4, 864, 54])
    235382
    
    
    bs = 4
    c_in = 5
    seq_len = 50
    
    encoder = _TransformerRNNEncoder(nn.LSTM, c_in=c_in, seq_len=seq_len, d_model=128, nhead=4, num_encoder_layers=1, dim_feedforward=None, proj_dropout=0.1, dropout=0.1, num_rnn_layers=3, bidirectional=True)
    t = torch.randn(bs, c_in, seq_len)
    print(encoder(t).shape)__
    
    
    torch.Size([4, 1024, 50])

* * *

source

### TransformerGRUPlus

> 
>      TransformerGRUPlus (c_in:int, c_out:int, seq_len:int, d:tuple=None,
>                          d_model:int=128, nhead:int=16,
>                          proj_dropout:float=0.1, num_encoder_layers:int=1,
>                          dim_feedforward:int=2048, dropout:float=0.1,
>                          num_rnn_layers:int=1, bidirectional:bool=True,
>                          custom_head=None, **kwargs)

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
c_in | int |  | Number of channels in the input tensor.  
c_out | int |  | Number of output channels.  
seq_len | int |  | Number of time steps in the input tensor.  
d | tuple | None | int or tuple with shape of the output tensor  
d_model | int | 128 | Total dimension of the model.  
nhead | int | 16 | Number of parallel attention heads (d_model will be split across nhead - each head will have dimension d_model // nhead).  
proj_dropout | float | 0.1 | Dropout probability after the first linear layer. Default: 0.1.  
num_encoder_layers | int | 1 | Number of transformer encoder layers. Default: 1.  
dim_feedforward | int | 2048 | The dimension of the feedforward network model. Default: 2048.  
dropout | float | 0.1 | Transformer encoder layers dropout. Default: 0.1.  
num_rnn_layers | int | 1 | Number of RNN layers in the encoder. Default: 1.  
bidirectional | bool | True | If True, becomes a bidirectional RNN. Default: True.  
custom_head | NoneType | None | Custom head that will be applied to the model. If None, a head with `c_out` outputs will be used. Default: None.  
kwargs | VAR_KEYWORD |  |   
  
* * *

source

### TransformerLSTMPlus

> 
>      TransformerLSTMPlus (c_in:int, c_out:int, seq_len:int, d:tuple=None,
>                           d_model:int=128, nhead:int=16,
>                           proj_dropout:float=0.1, num_encoder_layers:int=1,
>                           dim_feedforward:int=2048, dropout:float=0.1,
>                           num_rnn_layers:int=1, bidirectional:bool=True,
>                           custom_head=None, **kwargs)

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
c_in | int |  | Number of channels in the input tensor.  
c_out | int |  | Number of output channels.  
seq_len | int |  | Number of time steps in the input tensor.  
d | tuple | None | int or tuple with shape of the output tensor  
d_model | int | 128 | Total dimension of the model.  
nhead | int | 16 | Number of parallel attention heads (d_model will be split across nhead - each head will have dimension d_model // nhead).  
proj_dropout | float | 0.1 | Dropout probability after the first linear layer. Default: 0.1.  
num_encoder_layers | int | 1 | Number of transformer encoder layers. Default: 1.  
dim_feedforward | int | 2048 | The dimension of the feedforward network model. Default: 2048.  
dropout | float | 0.1 | Transformer encoder layers dropout. Default: 0.1.  
num_rnn_layers | int | 1 | Number of RNN layers in the encoder. Default: 1.  
bidirectional | bool | True | If True, becomes a bidirectional RNN. Default: True.  
custom_head | NoneType | None | Custom head that will be applied to the model. If None, a head with `c_out` outputs will be used. Default: None.  
kwargs | VAR_KEYWORD |  |   
  
* * *

source

### TransformerRNNPlus

> 
>      TransformerRNNPlus (c_in:int, c_out:int, seq_len:int, d:tuple=None,
>                          d_model:int=128, nhead:int=16,
>                          proj_dropout:float=0.1, num_encoder_layers:int=1,
>                          dim_feedforward:int=2048, dropout:float=0.1,
>                          num_rnn_layers:int=1, bidirectional:bool=True,
>                          custom_head=None, **kwargs)

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
c_in | int |  | Number of channels in the input tensor.  
c_out | int |  | Number of output channels.  
seq_len | int |  | Number of time steps in the input tensor.  
d | tuple | None | int or tuple with shape of the output tensor  
d_model | int | 128 | Total dimension of the model.  
nhead | int | 16 | Number of parallel attention heads (d_model will be split across nhead - each head will have dimension d_model // nhead).  
proj_dropout | float | 0.1 | Dropout probability after the first linear layer. Default: 0.1.  
num_encoder_layers | int | 1 | Number of transformer encoder layers. Default: 1.  
dim_feedforward | int | 2048 | The dimension of the feedforward network model. Default: 2048.  
dropout | float | 0.1 | Transformer encoder layers dropout. Default: 0.1.  
num_rnn_layers | int | 1 | Number of RNN layers in the encoder. Default: 1.  
bidirectional | bool | True | If True, becomes a bidirectional RNN. Default: True.  
custom_head | NoneType | None | Custom head that will be applied to the model. If None, a head with `c_out` outputs will be used. Default: None.  
kwargs | VAR_KEYWORD |  |   
      
    
    bs = 4
    c_in = 5
    c_out = 1
    seq_len = 50
    d = None
    
    model = TransformerRNNPlus(c_in=c_in, c_out=c_out, seq_len=seq_len, d=d, proj_dropout=0.1, d_model=128, nhead=4, num_encoder_layers=2, dropout=0.1, num_rnn_layers=1, bidirectional=True)
    t = torch.randn(bs, c_in, seq_len)
    assert model(t).shape == torch.Size([4]) 
    print(model(t).shape)
    
    model = TransformerLSTMPlus(c_in=c_in, c_out=c_out, seq_len=seq_len, d=d, proj_dropout=0.1, d_model=128, nhead=4, num_encoder_layers=2, dropout=0.1, num_rnn_layers=1, bidirectional=True)
    t = torch.randn(bs, c_in, seq_len)
    assert model(t).shape == torch.Size([4])
    print(model(t).shape)
    
    model = TransformerGRUPlus(c_in=c_in, c_out=c_out, seq_len=seq_len, d=d, proj_dropout=0.1, d_model=128, nhead=4, num_encoder_layers=2, dropout=0.1, num_rnn_layers=1, bidirectional=True)
    t = torch.randn(bs, c_in, seq_len)
    assert model(t).shape == torch.Size([4])
    print(model(t).shape)__
    
    
    torch.Size([4])
    torch.Size([4])
    torch.Size([4])
    
    
    bs = 4
    c_in = 5
    c_out = 3
    seq_len = 50
    d = None
    
    model = TransformerRNNPlus(c_in=c_in, c_out=c_out, seq_len=seq_len, d=d, proj_dropout=0.1, d_model=128, nhead=4, num_encoder_layers=2, dropout=0.1, num_rnn_layers=1, bidirectional=True)
    t = torch.randn(bs, c_in, seq_len)
    assert model(t).shape == (bs, c_out)
    print(model(t).shape)
    
    model = TransformerLSTMPlus(c_in=c_in, c_out=c_out, seq_len=seq_len, d=d, proj_dropout=0.1, d_model=128, nhead=4, num_encoder_layers=2, dropout=0.1, num_rnn_layers=1, bidirectional=True)
    t = torch.randn(bs, c_in, seq_len)
    assert model(t).shape == (bs, c_out)
    print(model(t).shape)
    
    model = TransformerGRUPlus(c_in=c_in, c_out=c_out, seq_len=seq_len, d=d, proj_dropout=0.1, d_model=128, nhead=4, num_encoder_layers=2, dropout=0.1, num_rnn_layers=1, bidirectional=True)
    t = torch.randn(bs, c_in, seq_len)
    assert model(t).shape == (bs, c_out)
    print(model(t).shape)__
    
    
    torch.Size([4, 3])
    torch.Size([4, 3])
    torch.Size([4, 3])
    
    
    bs = 4
    c_in = 5
    c_out = 3
    seq_len = 50
    d = 50
    
    model = TransformerRNNPlus(c_in=c_in, c_out=c_out, seq_len=seq_len, d=d, proj_dropout=0.1, d_model=128, nhead=4, num_encoder_layers=2, dropout=0.1, num_rnn_layers=1, bidirectional=True)
    t = torch.randn(bs, c_in, seq_len)
    assert model(t).shape == (bs, d, c_out)
    print(model(t).shape)
    
    model = TransformerLSTMPlus(c_in=c_in, c_out=c_out, seq_len=seq_len, d=d, proj_dropout=0.1, d_model=128, nhead=4, num_encoder_layers=2, dropout=0.1, num_rnn_layers=1, bidirectional=True)
    t = torch.randn(bs, c_in, seq_len)
    assert model(t).shape == (bs, d, c_out)
    print(model(t).shape)
    
    model = TransformerGRUPlus(c_in=c_in, c_out=c_out, seq_len=seq_len, d=d, proj_dropout=0.1, d_model=128, nhead=4, num_encoder_layers=2, dropout=0.1, num_rnn_layers=1, bidirectional=True)
    t = torch.randn(bs, c_in, seq_len)
    assert model(t).shape == (bs, d, c_out)
    print(model(t).shape)__
    
    
    torch.Size([4, 50, 3])
    torch.Size([4, 50, 3])
    torch.Size([4, 50, 3])

  * __Report an issue


