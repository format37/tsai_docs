## On this page

  * ROCKET
  * create_rocket_features



  * __Report an issue



  1. Models
  2. ROCKETs
  3. ROCKET Pytorch



# ROCKET Pytorch

> ROCKET (RandOm Convolutional KErnel Transform) functions for univariate and multivariate time series developed in Pytorch.

* * *

source

### ROCKET

> 
>      ROCKET (c_in, seq_len, n_kernels=10000, kss=[7, 9, 11], device=None,
>              verbose=False)

*RandOm Convolutional KErnel Transform

ROCKET is a GPU Pytorch implementation of the ROCKET functions generate_kernels and apply_kernels that can be used with univariate and multivariate time series.*

* * *

source

### create_rocket_features

> 
>      create_rocket_features (dl, model, verbose=False)

_Args: model : ROCKET model instance dl : single TSDataLoader (for example dls.train or dls.valid)_
    
    
    bs = 16
    c_in = 7  # aka channels, features, variables, dimensions
    c_out = 2
    seq_len = 15
    xb = torch.randn(bs, c_in, seq_len).to(default_device())
    
    m = ROCKET(c_in, seq_len, n_kernels=1_000, kss=[7, 9, 11]) # 1_000 for testing with a cpu. Default is 10k with a gpu!
    test_eq(m(xb).shape, [bs, 2_000])__
    
    
    from tsai.data.all import *
    from tsai.models.utils import *__
    
    
    X, y, splits = get_UCR_data('OliveOil', split_data=False)
    tfms = [None, TSRegression()]
    batch_tfms = TSStandardize(by_var=True)
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, shuffle_train=False, drop_last=False)
    model = build_ts_model(ROCKET, dls=dls, n_kernels=1_000) # 1_000 for testing with a cpu. Default is 10k with a gpu!
    X_train, y_train = create_rocket_features(dls.train, model) 
    X_valid, y_valid = create_rocket_features(dls.valid, model)
    X_train.shape, X_valid.shape __
    
    
    ((30, 2000), (30, 2000))

  * __Report an issue


