## On this page

  * Imports
  * TST
    * TSTPlus
    * MultiTSTPlus



  * __Report an issue



  1. Models
  2. Transformers
  3. TSTPlus



# TSTPlus

This is an unofficial PyTorch implementation by Ignacio Oguiza of - oguiza@timeseriesAI.co based on:

  * George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’21), August 14–18, 2021. ArXiV version: https://arxiv.org/abs/2010.02803
  * Official implementation: https://github.com/gzerveas/mvts_transformer


    
    
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

  * Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).

  * He, R., Ravula, A., Kanagal, B., & Ainslie, J. (2020). Realformer: Transformer Likes Informed Attention. arXiv preprint arXiv:2012.11747.




This implementation is adapted to work with the rest of the `tsai` library, and contain some hyperparameters that are not available in the original implementation. I included them for experimenting.

## Imports

## TST
    
    
    t = torch.rand(16, 50, 128)
    attn_mask = torch.triu(torch.ones(50, 50)) # shape: q_len x q_len
    key_padding_mask = torch.zeros(16, 50)
    key_padding_mask[[1, 3, 6, 15], -10:] = 1
    key_padding_mask = key_padding_mask.bool()
    print('attn_mask', attn_mask.shape, 'key_padding_mask', key_padding_mask.shape)
    encoder = _TSTEncoderLayer(q_len=50, d_model=128, n_heads=8, d_k=None, d_v=None, d_ff=512, attn_dropout=0., dropout=0.1, store_attn=True, activation='gelu')
    output = encoder(t, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
    output.shape __
    
    
    attn_mask torch.Size([50, 50]) key_padding_mask torch.Size([16, 50])
    
    
    torch.Size([16, 50, 128])
    
    
    cmap='viridis'
    figsize=(6,5)
    plt.figure(figsize=figsize)
    plt.pcolormesh(encoder.attn[0][0].detach().cpu().numpy(), cmap=cmap)
    plt.title('Self-attention map')
    plt.colorbar()
    plt.show()__

* * *

source

### TSTPlus

> 
>      TSTPlus (c_in:int, c_out:int, seq_len:int, max_seq_len:Optional[int]=512,
>               n_layers:int=3, d_model:int=128, n_heads:int=16,
>               d_k:Optional[int]=None, d_v:Optional[int]=None, d_ff:int=256,
>               norm:str='BatchNorm', attn_dropout:float=0.0, dropout:float=0.0,
>               act:str='gelu', key_padding_mask:bool='auto',
>               padding_var:Optional[int]=None,
>               attn_mask:Optional[torch.Tensor]=None, res_attention:bool=True,
>               pre_norm:bool=False, store_attn:bool=False, pe:str='zeros',
>               learn_pe:bool=True, flatten:bool=True, fc_dropout:float=0.0,
>               concat_pool:bool=False, bn:bool=False,
>               custom_head:Optional[Callable]=None,
>               y_range:Optional[tuple]=None, verbose:bool=False, **kwargs)

_TST (Time Series Transformer) is a Transformer that takes continuous time series as inputs_
    
    
    from tsai.models.utils import build_ts_model __
    
    
    bs = 8
    c_in = 9  # aka channels, features, variables, dimensions
    c_out = 2
    seq_len = 1_500
    
    xb = torch.randn(bs, c_in, seq_len).to(device)
    
    # standardize by channel by_var based on the training set
    xb = (xb - xb.mean((0, 2), keepdim=True)) / xb.std((0, 2), keepdim=True)
    
    # Settings
    max_seq_len = 256
    d_model = 128
    n_heads = 16
    d_k = d_v = None  # if None --> d_model // n_heads
    d_ff = 256
    norm = "BatchNorm"
    dropout = 0.1
    activation = "gelu"
    n_layers = 3
    fc_dropout = 0.1
    pe = None
    learn_pe = True
    kwargs = {}
    
    model = TSTPlus(c_in, c_out, seq_len, max_seq_len=max_seq_len, d_model=d_model, n_heads=n_heads,
                    d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, dropout=dropout, activation=activation, n_layers=n_layers,
                    fc_dropout=fc_dropout, pe=pe, learn_pe=learn_pe, **kwargs).to(device)
    test_eq(model(xb).shape, [bs, c_out])
    test_eq(model[0], model.backbone)
    test_eq(model[1], model.head)
    model2 = build_ts_model(TSTPlus, c_in, c_out, seq_len, max_seq_len=max_seq_len, d_model=d_model, n_heads=n_heads,
                               d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, dropout=dropout, activation=activation, n_layers=n_layers,
                               fc_dropout=fc_dropout, pe=pe, learn_pe=learn_pe, **kwargs).to(device)
    test_eq(model2(xb).shape, [bs, c_out])
    test_eq(model2[0], model2.backbone)
    test_eq(model2[1], model2.head)
    print(f'model parameters: {count_parameters(model)}')__
    
    
    model parameters: 470018
    
    
    key_padding_mask = torch.sort(torch.randint(0, 2, (bs, max_seq_len))).values.bool().to(device)
    key_padding_mask[0]__
    
    
    tensor([False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True])
    
    
    model2.key_padding_mask = True
    model2.to(device)((xb, key_padding_mask)).shape __
    
    
    torch.Size([8, 2])
    
    
    model.head __
    
    
    Sequential(
      (0): GELU(approximate='none')
      (1): fastai.layers.Flatten(full=False)
      (2): LinBnDrop(
        (0): Dropout(p=0.1, inplace=False)
        (1): Linear(in_features=32768, out_features=2, bias=True)
      )
    )
    
    
    model = TSTPlus(c_in, c_out, seq_len, pre_norm=True)
    test_eq(model.to(xb.device)(xb).shape, [bs, c_out])__
    
    
    bs = 8
    c_in = 9  # aka channels, features, variables, dimensions
    c_out = 2
    seq_len = 5000
    
    xb = torch.randn(bs, c_in, seq_len)
    
    # standardize by channel by_var based on the training set
    xb = (xb - xb.mean((0, 2), keepdim=True)) / xb.std((0, 2), keepdim=True)
    
    model = TSTPlus(c_in, c_out, seq_len, res_attention=True)
    test_eq(model.to(xb.device)(xb).shape, [bs, c_out])
    print(f'model parameters: {count_parameters(model)}')__
    
    
    model parameters: 605698
    
    
    custom_head = partial(create_pool_head, concat_pool=True)
    model = TSTPlus(c_in, c_out, seq_len, max_seq_len=max_seq_len, d_model=d_model, n_heads=n_heads,
                d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, activation=activation, n_layers=n_layers,
                fc_dropout=fc_dropout, pe=pe, learn_pe=learn_pe, flatten=False, custom_head=custom_head, **kwargs)
    test_eq(model.to(xb.device)(xb).shape, [bs, c_out])
    print(f'model parameters: {count_parameters(model)}')__
    
    
    model parameters: 421122
    
    
    custom_head = partial(create_pool_plus_head, concat_pool=True)
    model = TSTPlus(c_in, c_out, seq_len, max_seq_len=max_seq_len, d_model=d_model, n_heads=n_heads,
                d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, activation=activation, n_layers=n_layers,
                fc_dropout=fc_dropout, pe=pe, learn_pe=learn_pe, flatten=False, custom_head=custom_head, **kwargs)
    test_eq(model.to(xb.device)(xb).shape, [bs, c_out])
    print(f'model parameters: {count_parameters(model)}')__
    
    
    model parameters: 554240
    
    
    bs = 8
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
    pe='zeros'
    learn_pe=True
    kwargs = {}
    # kwargs = dict(kernel_size=5, padding=2)
    
    model = TSTPlus(c_in, c_out, seq_len, max_seq_len=max_seq_len, d_model=d_model, n_heads=n_heads,
                d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, act=act, n_layers=n_layers,
                fc_dropout=fc_dropout, pe=pe, learn_pe=learn_pe, **kwargs)
    test_eq(model.to(xb.device)(xb).shape, [bs, c_out])
    print(f'model parameters: {count_parameters(model)}')
    body, head = model[0], model[1]
    test_eq(body.to(xb.device)(xb).ndim, 3)
    test_eq(head.to(xb.device)(body.to(xb.device)(xb)).ndim, 2)
    head __
    
    
    model parameters: 421762
    
    
    Sequential(
      (0): GELU(approximate='none')
      (1): fastai.layers.Flatten(full=False)
      (2): LinBnDrop(
        (0): Dropout(p=0.1, inplace=False)
        (1): Linear(in_features=7680, out_features=2, bias=True)
      )
    )
    
    
    model.show_pe()__
    
    
    model = TSTPlus(3, 2, 10)
    xb = torch.randn(4, 3, 10)
    yb = torch.randint(0, 2, (4,))
    test_eq(model.backbone._key_padding_mask(xb)[1], None)
    random_idxs = random_choice(len(xb), 2, False)
    xb[random_idxs, :, -5:] = np.nan
    xb[random_idxs, 0, 1] = np.nan
    test_eq(model.backbone._key_padding_mask(xb.clone())[1].data, (torch.isnan(xb).float().mean(1)==1).bool())
    test_eq(model.backbone._key_padding_mask(xb.clone())[1].data.shape, (4,10))
    print(torch.isnan(xb).sum())
    pred = model.to(xb.device)(xb.clone())
    loss = CrossEntropyLossFlat()(pred, yb)
    loss.backward()
    model.to(xb.device).backbone._key_padding_mask(xb)[1].data.shape __
    
    
    tensor(32)
    
    
    torch.Size([4, 10])
    
    
    bs = 4
    c_in = 3
    seq_len = 10
    c_out = 2
    xb = torch.randn(bs, c_in, seq_len)
    xb[:, -1] = torch.randint(0, 2, (bs, seq_len)).sort()[0]
    model = TSTPlus(c_in, c_out, seq_len).to(xb.device)
    test_eq(model.backbone._key_padding_mask(xb)[1], None)
    model = TSTPlus(c_in, c_out, seq_len, padding_var=-1).to(xb.device)
    test_eq(model.backbone._key_padding_mask(xb)[1], (xb[:, -1]==1))
    model = TSTPlus(c_in, c_out, seq_len, padding_var=2).to(xb.device)
    test_eq(model.backbone._key_padding_mask(xb)[1], (xb[:, -1]==1))
    test_eq(model(xb).shape, (bs, c_out))__
    
    
    bs = 4
    c_in = 3
    seq_len = 10
    c_out = 2
    xb = torch.randn(bs, c_in, seq_len)
    model = TSTPlus(c_in, c_out, seq_len, act='smelu')__

* * *

source

### MultiTSTPlus

> 
>      MultiTSTPlus (feat_list, c_out, seq_len, max_seq_len:Optional[int]=512,
>                    custom_head=None, n_layers:int=3, d_model:int=128,
>                    n_heads:int=16, d_k:Optional[int]=None,
>                    d_v:Optional[int]=None, d_ff:int=256, norm:str='BatchNorm',
>                    attn_dropout:float=0.0, dropout:float=0.0, act:str='gelu',
>                    key_padding_mask:bool='auto',
>                    padding_var:Optional[int]=None,
>                    attn_mask:Optional[torch.Tensor]=None,
>                    res_attention:bool=True, pre_norm:bool=False,
>                    store_attn:bool=False, pe:str='zeros', learn_pe:bool=True,
>                    flatten:bool=True, fc_dropout:float=0.0,
>                    concat_pool:bool=False, bn:bool=False,
>                    y_range:Optional[tuple]=None, verbose:bool=False)

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
    
    
    bs = 8
    c_in = 7  # aka channels, features, variables, dimensions
    c_out = 2
    seq_len = 10
    xb2 = torch.randn(bs, c_in, seq_len)
    model1 = MultiTSTPlus([2, 5], c_out, seq_len)
    model2 = MultiTSTPlus(7, c_out, seq_len)
    test_eq(model1.to(xb2.device)(xb2).shape, (bs, c_out))
    test_eq(model1.to(xb2.device)(xb2).shape, model2.to(xb2.device)(xb2).shape)
    test_eq(count_parameters(model1) > count_parameters(model2), True)__
    
    
    bs = 8
    c_in = 7  # aka channels, features, variables, dimensions
    c_out = 2
    seq_len = 10
    xb2 = torch.randn(bs, c_in, seq_len)
    model1 = MultiTSTPlus([2, 5], c_out, seq_len, )
    model2 = MultiTSTPlus([[0,2,5], [0,1,3,4,6]], c_out, seq_len)
    test_eq(model1.to(xb2.device)(xb2).shape, (bs, c_out))
    test_eq(model1.to(xb2.device)(xb2).shape, model2.to(xb2.device)(xb2).shape)__
    
    
    model1 = MultiTSTPlus([2, 5], c_out, seq_len, y_range=(0.5, 5.5))
    body, head = split_model(model1)
    test_eq(body.to(xb2.device)(xb2).ndim, 3)
    test_eq(head.to(xb2.device)(body.to(xb2.device)(xb2)).ndim, 2)
    head __
    
    
    Sequential(
      (0): Sequential(
        (0): GELU(approximate='none')
        (1): fastai.layers.Flatten(full=False)
        (2): LinBnDrop(
          (0): Linear(in_features=2560, out_features=2, bias=True)
        )
      )
    )
    
    
    model = MultiTSTPlus([2, 5], c_out, seq_len, pre_norm=True)__
    
    
    bs = 8
    n_vars = 3
    seq_len = 12
    c_out = 2
    xb = torch.rand(bs, n_vars, seq_len)
    net = MultiTSTPlus(n_vars, c_out, seq_len)
    change_model_head(net, create_pool_plus_head, concat_pool=False)
    print(net.to(xb.device)(xb).shape)
    net.head __
    
    
    torch.Size([8, 2])
    
    
    Sequential(
      (0): AdaptiveAvgPool1d(output_size=1)
      (1): Reshape(bs)
      (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): Linear(in_features=128, out_features=512, bias=False)
      (4): ReLU(inplace=True)
      (5): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (6): Linear(in_features=512, out_features=2, bias=False)
    )
    
    
    bs = 8
    n_vars = 3
    seq_len = 12
    c_out = 10
    xb = torch.rand(bs, n_vars, seq_len)
    new_head = partial(conv_lin_nd_head, d=(5 ,2))
    net = MultiTSTPlus(n_vars, c_out, seq_len, custom_head=new_head)
    print(net.to(xb.device)(xb).shape)
    net.head __
    
    
    torch.Size([8, 5, 2, 10])
    
    
    Sequential(
      (0): create_conv_lin_nd_head(
        (0): Conv1d(128, 10, kernel_size=(1,), stride=(1,))
        (1): Linear(in_features=12, out_features=10, bias=True)
        (2): Transpose(-1, -2)
        (3): Reshape(bs, 5, 2, 10)
      )
    )

  * __Report an issue


