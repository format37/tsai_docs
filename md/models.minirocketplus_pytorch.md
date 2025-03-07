## On this page

  * MiniRocketFeaturesPlus
  * MiniRocketPlus
  * Flatten
  * get_minirocket_features
  * MiniRocketHead
  * InceptionRocketFeaturesPlus
  * InceptionRocketPlus



  * __Report an issue



  1. Models
  2. ROCKETs
  3. MINIROCKETPlus Pytorch



# MINIROCKETPlus Pytorch

This is a modified Pytorch implementation of MiniRocket originally developed by Malcolm McLean and Ignacio Oguiza and based on:

Dempster, A., Schmidt, D. F., & Webb, G. I. (2020). **MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification**. arXiv preprint arXiv:2012.08791.

Original paper: https://arxiv.org/abs/2012.08791

Original code: https://github.com/angus924/minirocket

* * *

source

### MiniRocketFeaturesPlus

> 
>      MiniRocketFeaturesPlus (c_in, seq_len, num_features=10000,
>                              max_dilations_per_kernel=32, kernel_size=9,
>                              max_num_channels=9, max_num_kernels=84,
>                              add_lsaz=False)

*Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes::
    
    
    import torch.nn as nn
    import torch.nn.functional as F
    
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)
    
        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will have their parameters converted too when you call :meth:`to`, etc.

.. note:: As per the example above, an `__init__()` call to the parent class must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or evaluation mode. :vartype training: bool*

* * *

source

### MiniRocketPlus

> 
>      MiniRocketPlus (c_in, c_out, seq_len, num_features=10000,
>                      max_dilations_per_kernel=32, kernel_size=9,
>                      max_num_channels=None, max_num_kernels=84, bn=True,
>                      fc_dropout=0, add_lsaz=False, custom_head=None,
>                      zero_init=True)

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

### Flatten

> 
>      Flatten (*args, **kwargs)

*Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes::
    
    
    import torch.nn as nn
    import torch.nn.functional as F
    
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)
    
        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will have their parameters converted too when you call :meth:`to`, etc.

.. note:: As per the example above, an `__init__()` call to the parent class must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or evaluation mode. :vartype training: bool*

* * *

source

### get_minirocket_features

> 
>      get_minirocket_features (o, model, chunksize=1024, use_cuda=None,
>                               to_np=False)

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
    
    
    from tsai.imports import default_device
    from fastai.metrics import accuracy
    from fastai.callback.tracker import ReduceLROnPlateau
    from tsai.data.all import *
    from tsai.learner import *__
    
    
    # Offline feature calculation
    dsid = 'ECGFiveDays'
    X, y, splits = get_UCR_data(dsid, split_data=False)
    mrf = MiniRocketFeaturesPlus(c_in=X.shape[1], seq_len=X.shape[2]).to(default_device())
    X_train = X[splits[0]]  # X_train may either be a np.ndarray or a torch.Tensor
    mrf.fit(X_train)
    X_tfm = get_minirocket_features(X, mrf).cpu().numpy()
    tfms = [None, TSClassification()]
    batch_tfms = TSStandardize(by_var=True)
    dls = get_ts_dls(X_tfm, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=256)
    learn = ts_learner(dls, MiniRocketHead, metrics=accuracy)
    learn.fit(1, 1e-4, cbs=ReduceLROnPlateau(factor=0.5, min_lr=1e-8, patience=10))__
    
    
    # Online feature calculation
    dsid = 'ECGFiveDays'
    X, y, splits = get_UCR_data(dsid, split_data=False)
    tfms = [None, TSClassification()]
    batch_tfms = TSStandardize()
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=256)
    learn = ts_learner(dls, MiniRocketPlus, kernel_size=7, metrics=accuracy)
    learn.fit_one_cycle(1, 1e-2)__
    
    
    from functools import partial
    from fastcore.test import *
    from tsai.models.utils import build_ts_model
    from tsai.models.layers import mlp_head, rocket_nd_head __
    
    
    bs, c_in, seq_len = 8, 3, 50
    c_out = 2
    xb = torch.randn(bs, c_in, seq_len)
    model = build_ts_model(MiniRocketPlus, c_in=c_in, c_out=c_out, seq_len=seq_len)
    test_eq(model.to(xb.device)(xb).shape, (bs, c_out))
    model = build_ts_model(MiniRocketPlus, c_in=c_in, c_out=c_out, seq_len=seq_len, add_lsaz=True)
    test_eq(model.to(xb.device)(xb).shape, (bs, c_out))
    model = build_ts_model(MiniRocketPlus, c_in=c_in, c_out=c_out, seq_len=seq_len, custom_head=mlp_head)
    test_eq(model.to(xb.device)(xb).shape, (bs, c_out))__
    
    
    X = np.random.rand(8, 10, 100)
    y = np.random.rand(8, 1, 100)
    splits = TimeSplitter(show_plot=False)(y)
    tfms = [None, TSRegression()]
    batch_tfms = TSStandardize(by_sample=True)
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
    custom_head = partial(rocket_nd_head, d=dls.d)
    model = MiniRocketPlus(dls.vars, dls.c, dls.len, custom_head=custom_head)
    xb,yb = dls.one_batch()
    test_eq(model.to(xb.device)(xb).shape[1:], y.shape[1:])__
    
    
    X = np.random.rand(16, 10, 100)
    y = np.random.randint(0, 4, (16, 1, 100))
    splits = TimeSplitter(show_plot=False)(y)
    tfms = [None, TSClassification()]
    batch_tfms = TSStandardize(by_sample=True)
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
    custom_head = partial(rocket_nd_head, d=dls.d)
    model = MiniRocketPlus(dls.vars, dls.c, dls.len, custom_head=custom_head)
    xb,yb = dls.one_batch()
    test_eq(model.to(xb.device)(xb).shape[1:], y.shape[1:]+(4,))__

* * *

source

### InceptionRocketFeaturesPlus

> 
>      InceptionRocketFeaturesPlus (c_in, seq_len, num_features=10000,
>                                   max_dilations_per_kernel=32,
>                                   kernel_sizes=array([3, 5, 7, 9]),
>                                   max_num_channels=None, max_num_kernels=84,
>                                   add_lsaz=True, same_n_feats_per_ks=False)

*Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes::
    
    
    import torch.nn as nn
    import torch.nn.functional as F
    
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)
    
        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will have their parameters converted too when you call :meth:`to`, etc.

.. note:: As per the example above, an `__init__()` call to the parent class must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or evaluation mode. :vartype training: bool*

* * *

source

### InceptionRocketPlus

> 
>      InceptionRocketPlus (c_in, c_out, seq_len, num_features=10000,
>                           max_dilations_per_kernel=32, kernel_sizes=[3, 5, 7,
>                           9], max_num_channels=None, max_num_kernels=84,
>                           same_n_feats_per_ks=False, add_lsaz=False, bn=True,
>                           fc_dropout=0, custom_head=None, zero_init=True)

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
    
    
    from fastcore.test import *
    from tsai.models.utils import build_ts_model __
    
    
    bs, c_in, seq_len = 8, 3, 50
    c_out = 2
    xb = torch.randn(bs, c_in, seq_len)
    model = build_ts_model(InceptionRocketPlus, c_in=c_in, c_out=c_out, seq_len=seq_len)
    test_eq(model.to(xb.device)(xb).shape, (bs, c_out))
    model = build_ts_model(InceptionRocketPlus, c_in=c_in, c_out=c_out, seq_len=seq_len, add_lsaz=True)
    test_eq(model.to(xb.device)(xb).shape, (bs, c_out))__
    
    
    X = np.random.rand(8, 10, 100)
    y = np.random.rand(8, 1, 100)
    splits = TimeSplitter(show_plot=False)(y)
    tfms = [None, TSRegression()]
    batch_tfms = TSStandardize(by_sample=True)
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
    custom_head = partial(rocket_nd_head, d=dls.d)
    model = InceptionRocketPlus(dls.vars, dls.c, dls.len, custom_head=custom_head)
    xb,yb = dls.one_batch()
    test_eq(model.to(xb.device)(xb).shape[1:], y.shape[1:])__
    
    
    X = np.random.rand(16, 10, 100)
    y = np.random.randint(0, 4, (16, 1, 100))
    splits = TimeSplitter(show_plot=False)(y)
    tfms = [None, TSClassification()]
    batch_tfms = TSStandardize(by_sample=True)
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
    custom_head = partial(rocket_nd_head, d=dls.d)
    model = MiniRocketPlus(dls.vars, dls.c, dls.len, custom_head=custom_head)
    xb,yb = dls.one_batch()
    test_eq(model.to(xb.device)(xb).shape[1:], y.shape[1:]+(4,))__

  * __Report an issue


