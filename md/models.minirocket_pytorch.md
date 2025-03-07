## On this page

  * MiniRocketFeatures
  * get_minirocket_features
  * MiniRocketHead
  * MiniRocket



  * __Report an issue



  1. Models
  2. ROCKETs
  3. MINIROCKET Pytorch



# MINIROCKET Pytorch

> A Very Fast (Almost) Deterministic Transform for Time Series Classification.

This is a Pytorch implementation of MiniRocket developed by Malcolm McLean and Ignacio Oguiza based on:

Dempster, A., Schmidt, D. F., & Webb, G. I. (2020). MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification. arXiv preprint arXiv:2012.08791.

Original paper: https://arxiv.org/abs/2012.08791

Original code: https://github.com/angus924/minirocket

* * *

source

### MiniRocketFeatures

> 
>      MiniRocketFeatures (c_in, seq_len, num_features=10000,
>                          max_dilations_per_kernel=32, random_state=None)

*This is a Pytorch implementation of MiniRocket developed by Malcolm McLean and Ignacio Oguiza

MiniRocket paper citation: @article{dempster_etal_2020, author = {Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I}, title = {{MINIROCKET}: A Very Fast (Almost) Deterministic Transform for Time Series Classification}, year = {2020}, journal = {arXiv:2012.08791} } Original paper: https://arxiv.org/abs/2012.08791 Original code: https://github.com/angus924/minirocket*

* * *

source

### get_minirocket_features

> 
>      get_minirocket_features (o, model, chunksize=1024, use_cuda=None,
>                               to_np=True)

_Function used to split a large dataset into chunks, avoiding OOM error._

* * *

source

### MiniRocketHead

> 
>      MiniRocketHead (c_in, c_out, seq_len=1, bn=True, fc_dropout=0.0)

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

* * *

source

### MiniRocket

> 
>      MiniRocket (c_in, c_out, seq_len, num_features=10000,
>                  max_dilations_per_kernel=32, random_state=None, bn=True,
>                  fc_dropout=0)

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
    
    
    from tsai.imports import default_device
    from fastai.metrics import accuracy
    from fastai.callback.tracker import ReduceLROnPlateau
    from tsai.data.all import *
    from tsai.learner import *__
    
    
    # Offline feature calculation
    dsid = 'ECGFiveDays'
    X, y, splits = get_UCR_data(dsid, split_data=False)
    mrf = MiniRocketFeatures(c_in=X.shape[1], seq_len=X.shape[2]).to(default_device())
    X_train = X[splits[0]]  # X_train may either be a np.ndarray or a torch.Tensor
    mrf.fit(X_train)
    X_tfm = get_minirocket_features(X, mrf)
    tfms = [None, TSClassification()]
    batch_tfms = TSStandardize(by_var=True)
    dls = get_ts_dls(X_tfm, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=256)
    learn = ts_learner(dls, MiniRocketHead, metrics=accuracy)
    learn.fit(1, 1e-4, cbs=ReduceLROnPlateau(factor=0.5, min_lr=1e-8, patience=10))__

epoch | train_loss | valid_loss | accuracy | time  
---|---|---|---|---  
0 | 0.693147 | 0.530879 | 0.752613 | 00:00  
      
    
    # Online feature calculation
    dsid = 'ECGFiveDays'
    X, y, splits = get_UCR_data(dsid, split_data=False)
    tfms = [None, TSClassification()]
    batch_tfms = TSStandardize()
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=256)
    learn = ts_learner(dls, MiniRocket, metrics=accuracy)
    learn.fit_one_cycle(1, 1e-2)__

epoch | train_loss | valid_loss | accuracy | time  
---|---|---|---|---  
0 | 0.693147 | 0.713297 | 0.502904 | 00:06  
  
  * __Report an issue


