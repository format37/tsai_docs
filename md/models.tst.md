## On this page

  * TST arguments
  * Imports
  * TST
    * TST



  * __Report an issue



  1. Models
  2. Transformers
  3. TST



# TST

This is an unofficial PyTorch implementation by Ignacio Oguiza of - oguiza@timeseriesAI.co based on: * George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’21), August 14–18, 2021. ArXiV version: https://arxiv.org/abs/2010.02803 * Official implementation: https://github.com/gzerveas/mvts_transformer
    
    
    @inproceedings{10.1145/3447548.3467401,
    author = {Zerveas, George and Jayaraman, Srideepika and Patel, Dhaval and Bhamidipaty, Anuradha and Eickhoff, Carsten},
    title = {A Transformer-Based Framework for Multivariate Time Series Representation Learning},
    year = {2021},
    isbn = {9781450383325},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3447548.3467401},
    doi = {10.1145/3447548.3467401},
    booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery &amp; Data Mining},
    pages = {2114–2124},
    numpages = {11},
    keywords = {regression, framework, multivariate time series, classification, transformer, deep learning, self-supervised learning, unsupervised learning, imputation},
    location = {Virtual Event, Singapore},
    series = {KDD '21}
    }__

This paper uses ‘Attention is all you need’ as a major reference: * Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … & Polosukhin, I. (2017). **Attention is all you need**. In Advances in neural information processing systems (pp. 5998-6008).

This implementation is adapted to work with the rest of the `tsai` library, and contain some hyperparameters that are not available in the original implementation. They are included to experiment with them.

## TST arguments

Usual values are the ones that appear in the “Attention is all you need” and “A Transformer-based Framework for Multivariate Time Series Representation Learning” papers.

The default values are the ones selected as a default configuration in the latter.

  * c_in: the number of features (aka variables, dimensions, channels) in the time series dataset. dls.var
  * c_out: the number of target classes. dls.c
  * seq_len: number of time steps in the time series. dls.len
  * max_seq_len: useful to control the temporal resolution in long time series to avoid memory issues. Default. None.
  * d_model: total dimension of the model (number of features created by the model). Usual values: 128-1024. Default: 128.
  * n_heads: parallel attention heads. Usual values: 8-16. Default: 16.
  * d_k: size of the learned linear projection of queries and keys in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
  * d_v: size of the learned linear projection of values in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
  * d_ff: the dimension of the feedforward network model. Usual values: 256-4096. Default: 256.
  * dropout: amount of residual dropout applied in the encoder. Usual values: 0.-0.3. Default: 0.1.
  * activation: the activation function of intermediate layer, relu or gelu. Default: ‘gelu’.
  * n_layers: the number of sub-encoder-layers in the encoder. Usual values: 2-8. Default: 3.
  * fc_dropout: dropout applied to the final fully connected layer. Usual values: 0.-0.8. Default: 0.
  * y_range: range of possible y values (used in regression tasks). Default: None
  * kwargs: nn.Conv1d kwargs. If not {}, a nn.Conv1d with those kwargs will be applied to original time series.



## Imports

## TST
    
    
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

### TST

> 
>      TST (c_in:int, c_out:int, seq_len:int, max_seq_len:Optional[int]=None,
>           n_layers:int=3, d_model:int=128, n_heads:int=16,
>           d_k:Optional[int]=None, d_v:Optional[int]=None, d_ff:int=256,
>           dropout:float=0.1, act:str='gelu', fc_dropout:float=0.0,
>           y_range:Optional[tuple]=None, verbose:bool=False, **kwargs)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    bs = 32
    c_in = 9  # aka channels, features, variables, dimensions
    c_out = 2
    seq_len = 5000
    
    xb = torch.randn(bs, c_in, seq_len)
    
    # standardize by channel by_var based on the training set
    xb = (xb - xb.mean((0, 2), keepdim=True)) / xb.std((0, 2), keepdim=True)
    
    # Settings
    max_seq_len = 256
    d_model = 128
    n_heads = 16
    d_k = d_v = None # if None --> d_model // n_heads
    d_ff = 256
    dropout = 0.1
    activation = "gelu"
    n_layers = 3
    fc_dropout = 0.1
    kwargs = {}
    
    model = TST(c_in, c_out, seq_len, max_seq_len=max_seq_len, d_model=d_model, n_heads=n_heads,
                d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, activation=activation, n_layers=n_layers,
                fc_dropout=fc_dropout, **kwargs)
    test_eq(model.to(xb.device)(xb).shape, [bs, c_out])
    print(f'model parameters: {count_parameters(model)}')__
    
    
    model parameters: 517378
    
    
    bs = 32
    c_in = 9  # aka channels, features, variables, dimensions
    c_out = 2
    seq_len = 60
    
    xb = torch.randn(bs, c_in, seq_len)
    
    # standardize by channel by_var based on the training set
    xb = (xb - xb.mean((0, 2), keepdim=True)) / xb.std((0, 2), keepdim=True)
    
    # Settings
    max_seq_len = 120
    d_model = 128
    n_heads = 16
    d_k = d_v = None # if None --> d_model // n_heads
    d_ff = 256
    dropout = 0.1
    act = "gelu"
    n_layers = 3
    fc_dropout = 0.1
    kwargs = {}
    # kwargs = dict(kernel_size=5, padding=2)
    
    model = TST(c_in, c_out, seq_len, max_seq_len=max_seq_len, d_model=d_model, n_heads=n_heads,
                d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, act=act, n_layers=n_layers,
                fc_dropout=fc_dropout, **kwargs)
    test_eq(model.to(xb.device)(xb).shape, [bs, c_out])
    print(f'model parameters: {count_parameters(model)}')__
    
    
    model parameters: 420226

  * __Report an issue


