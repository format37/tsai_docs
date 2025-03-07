## On this page

  * WaveBlock
  * mWDNPlus
  * mWDNBlocks
  * mWDN



  * __Report an issue



  1. Models
  2. Wavelet-based NNs
  3. mWDN



# mWDN

> multilevel Wavelet Decomposition Network (mWDN)

This is an unofficial PyTorch implementation created by Ignacio Oguiza - oguiza@timeseriesAI.co

* * *

source

### WaveBlock

> 
>      WaveBlock (c_in, c_out, seq_len, wavelet=None)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### mWDNPlus

> 
>      mWDNPlus (c_in, c_out, seq_len, d=None, levels=3, wavelet=None,
>                base_model=None, base_arch=<class
>                'tsai.models.InceptionTimePlus.InceptionTimePlus'>, **kwargs)

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

### mWDNBlocks

> 
>      mWDNBlocks (c_in, c_out, seq_len, levels=3, wavelet=None)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### mWDN

> 
>      mWDN (c_in, c_out, seq_len, levels=3, wavelet=None, base_arch=<class
>            'tsai.models.InceptionTimePlus.InceptionTimePlus'>, **kwargs)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    from tsai.models.TSTPlus import TSTPlus __
    
    
    bs = 16
    c_in = 3
    seq_len = 12
    c_out = 2
    xb = torch.rand(bs, c_in, seq_len).to(default_device())
    test_eq(mWDN(c_in, c_out, seq_len).to(xb.device)(xb).shape, [bs, c_out])
    model = mWDNPlus(c_in, c_out, seq_len, fc_dropout=.5)
    test_eq(model.to(xb.device)(xb).shape, [bs, c_out])
    model = mWDNPlus(c_in, c_out, seq_len, base_arch=TSTPlus, fc_dropout=.5)
    test_eq(model.to(xb.device)(xb).shape, [bs, c_out])__
    
    
    model.head, model.head_nf __
    
    
    (Sequential(
       (0): GELU(approximate='none')
       (1): fastai.layers.Flatten(full=False)
       (2): LinBnDrop(
         (0): Dropout(p=0.5, inplace=False)
         (1): Linear(in_features=1536, out_features=2, bias=True)
       )
     ),
     128)
    
    
    bs = 16
    c_in = 3
    seq_len = 12
    d = 10
    c_out = 2
    xb = torch.rand(bs, c_in, seq_len).to(default_device())
    model = mWDNPlus(c_in, c_out, seq_len, d=d)
    test_eq(model.to(xb.device)(xb).shape, [bs, d, c_out])__
    
    
    bs = 16
    c_in = 3
    seq_len = 12
    d = (5, 2)
    c_out = 2
    xb = torch.rand(bs, c_in, seq_len).to(default_device())
    model = mWDNPlus(c_in, c_out, seq_len, d=d)
    test_eq(model.to(xb.device)(xb).shape, [bs, *d, c_out])__

  * __Report an issue


