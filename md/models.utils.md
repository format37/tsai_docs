## On this page

  * apply_idxs
  * SeqTokenizer
  * get_embed_size
  * has_weight_or_bias
  * has_weight
  * has_bias
  * is_conv
  * is_affine_layer
  * is_conv_linear
  * is_bn
  * is_linear
  * is_layer
  * get_layers
  * check_weight
  * check_bias
  * get_nf
  * ts_splitter
  * transfer_weights
  * build_ts_model
  * count_parameters
  * build_tsimage_model
  * build_tabular_model
  * get_clones
  * split_model
  * output_size_calculator
  * change_model_head
  * true_forecaster
  * naive_forecaster



  * __Report an issue



  1. Models
  2. Model utilities



# Model utilities

> Utility functions used to build PyTorch timeseries models.

* * *

source

### apply_idxs

> 
>      apply_idxs (o, idxs)

_Function to apply indices to zarr, dask and numpy arrays_

* * *

source

### SeqTokenizer

> 
>      SeqTokenizer (c_in, embed_dim, token_size=60, norm=False)

_Generates non-overlapping tokens from sub-sequences within a sequence by applying a sliding window_

* * *

source

### get_embed_size

> 
>      get_embed_size (n_cat, rule='log2')
    
    
    test_eq(get_embed_size(35), 6)__

* * *

source

### has_weight_or_bias

> 
>      has_weight_or_bias (l)

* * *

source

### has_weight

> 
>      has_weight (l)

* * *

source

### has_bias

> 
>      has_bias (l)

* * *

source

### is_conv

> 
>      is_conv (l)

* * *

source

### is_affine_layer

> 
>      is_affine_layer (l)

* * *

source

### is_conv_linear

> 
>      is_conv_linear (l)

* * *

source

### is_bn

> 
>      is_bn (l)

* * *

source

### is_linear

> 
>      is_linear (l)

* * *

source

### is_layer

> 
>      is_layer (*args)

* * *

source

### get_layers

> 
>      get_layers (model, cond=<function noop>, full=True)

* * *

source

### check_weight

> 
>      check_weight (m, cond=<function noop>, verbose=False)

* * *

source

### check_bias

> 
>      check_bias (m, cond=<function noop>, verbose=False)

* * *

source

### get_nf

> 
>      get_nf (m)

_Get nf from model’s first linear layer in head_

* * *

source

### ts_splitter

> 
>      ts_splitter (m)

_Split of a model between body and head_

* * *

source

### transfer_weights

> 
>      transfer_weights (model, weights_path:pathlib.Path,
>                        device:torch.device=None, exclude_head:bool=True)

_Utility function that allows to easily transfer weights between models. Taken from the great self-supervised repository created by Kerem Turgutlu. https://github.com/KeremTurgutlu/self_supervised/blob/d87ebd9b4961c7da0efd6073c42782bbc61aaa2e/self_supervised/utils.py_

* * *

source

### build_ts_model

> 
>      build_ts_model (arch, c_in=None, c_out=None, seq_len=None, d=None,
>                      dls=None, device=None, verbose=False, s_cat_idxs=None,
>                      s_cat_embeddings=None, s_cat_embedding_dims=None,
>                      s_cont_idxs=None, o_cat_idxs=None, o_cat_embeddings=None,
>                      o_cat_embedding_dims=None, o_cont_idxs=None,
>                      patch_len=None, patch_stride=None, fusion_layers=128,
>                      fusion_act='relu', fusion_dropout=0.0,
>                      fusion_use_bn=True, pretrained=False, weights_path=None,
>                      exclude_head=True, cut=-1, init=None, arch_config={},
>                      **kwargs)
    
    
    from tsai.data.core import get_ts_dls, TSClassification
    from tsai.models.TSiTPlus import TSiTPlus
    from fastai.losses import CrossEntropyLossFlat __
    
    
    X = np.random.rand(16, 3, 128)
    y = np.random.randint(0, 2, (16, 3))
    tfms = [None, [TSClassification()]]
    dls = get_ts_dls(X, y, splits=RandomSplitter()(range_of(X)), tfms=tfms)
    model = build_ts_model(TSiTPlus, dls=dls, pretrained=False, verbose=True)
    xb, yb = dls.one_batch()
    output = model(xb)
    print(output.shape)
    loss = CrossEntropyLossFlat()(output, yb)
    print(loss)
    assert output.shape == (dls.bs, dls.d, dls.c)__
    
    
    arch: TSiTPlus(c_in=3 c_out=2 seq_len=128 arch_config={} kwargs={'custom_head': functools.partial(<class 'tsai.models.layers.lin_nd_head'>, d=3)})
    torch.Size([13, 3, 2])
    TensorBase(0.8796, grad_fn=<AliasBackward0>)

* * *

source

### count_parameters

> 
>      count_parameters (model, trainable=True)

* * *

source

### build_tsimage_model

> 
>      build_tsimage_model (arch, c_in=None, c_out=None, dls=None,
>                           pretrained=False, device=None, verbose=False,
>                           init=None, arch_config={}, **kwargs)

* * *

source

### build_tabular_model

> 
>      build_tabular_model (arch, dls, layers=None, emb_szs=None, n_out=None,
>                           y_range=None, device=None, arch_config={}, **kwargs)
    
    
    from tsai.data.external import get_UCR_data
    from tsai.data.core import TSCategorize, get_ts_dls
    from tsai.data.preprocessing import TSStandardize
    from tsai.models.InceptionTime import *__
    
    
    X, y, splits = get_UCR_data('NATOPS', split_data=False)
    tfms = [None, TSCategorize()]
    batch_tfms = TSStandardize()
    dls = get_ts_dls(X, y, splits, tfms=tfms, batch_tfms=batch_tfms)
    model = build_ts_model(InceptionTime, dls=dls)
    test_eq(count_parameters(model), 460038)__

* * *

source

### get_clones

> 
>      get_clones (module, N)
    
    
    m = nn.Conv1d(3,4,3)
    get_clones(m, 3)__
    
    
    ModuleList(
      (0-2): 3 x Conv1d(3, 4, kernel_size=(3,), stride=(1,))
    )

* * *

source

### split_model

> 
>      split_model (m)

* * *

source

### output_size_calculator

> 
>      output_size_calculator (mod, c_in, seq_len=None)
    
    
    c_in = 3
    seq_len = 30
    m = nn.Conv1d(3, 12, kernel_size=3, stride=2)
    new_c_in, new_seq_len = output_size_calculator(m, c_in, seq_len)
    test_eq((new_c_in, new_seq_len), (12, 14))__

* * *

source

### change_model_head

> 
>      change_model_head (model, custom_head, **kwargs)

_Replaces a model’s head by a custom head as long as the model has a head, head_nf, c_out and seq_len attributes_

* * *

source

### true_forecaster

> 
>      true_forecaster (o, split, horizon=1)

* * *

source

### naive_forecaster

> 
>      naive_forecaster (o, split, horizon=1)
    
    
    a = np.random.rand(20).cumsum()
    split = np.arange(10, 20)
    a, naive_forecaster(a, split, 1), true_forecaster(a, split, 1)__
    
    
    (array([0.99029138, 1.68463991, 2.21744589, 2.65448222, 2.85159354,
            3.26171729, 3.67986707, 4.04343956, 4.3077458 , 4.44585435,
            4.76876866, 4.85844441, 4.93256093, 5.52415845, 6.10704489,
            6.74848957, 7.31920741, 8.20198208, 8.78954283, 9.0402    ]),
     array([4.44585435, 4.76876866, 4.85844441, 4.93256093, 5.52415845,
            6.10704489, 6.74848957, 7.31920741, 8.20198208, 8.78954283]),
     array([4.76876866, 4.85844441, 4.93256093, 5.52415845, 6.10704489,
            6.74848957, 7.31920741, 8.20198208, 8.78954283, 9.0402    ]))

  * __Report an issue


