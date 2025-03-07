## On this page

  * XResNet1dPlus
  * xresnet1d50_deeperplus
  * xresnet1d34_deeperplus
  * xresnet1d18_deeperplus
  * xresnet1d50_deepplus
  * xresnet1d34_deepplus
  * xresnet1d18_deepplus
  * xresnet1d152plus
  * xresnet1d101plus
  * xresnet1d50plus
  * xresnet1d34plus
  * xresnet1d18plus



  * __Report an issue



  1. Models
  2. CNNs
  3. XResNet1dPlus



# XResNet1dPlus

> This is a modified version of fastai’s XResNet model in github

* * *

source

### XResNet1dPlus

> 
>      XResNet1dPlus (block=<class 'tsai.models.layers.ResBlock1dPlus'>,
>                     expansion=4, layers=[3, 4, 6, 3], fc_dropout=0.0, c_in=3,
>                     c_out=None, n_out=1000, seq_len=None, stem_szs=(32, 32,
>                     64), widen=1.0, sa=False, act_cls=<class
>                     'torch.nn.modules.activation.ReLU'>, ks=3, stride=2,
>                     coord=False, custom_head=None, block_szs_base=(64, 128,
>                     256, 512), groups=1, reduction=None, nh1=None, nh2=None,
>                     dw=False, g2=1, sym=False, norm='Batch', zero_norm=True,
>                     pool=<function AvgPool>, pool_first=True)

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

### xresnet1d50_deeperplus

> 
>      xresnet1d50_deeperplus (c_in, c_out, seq_len=None, act=<class
>                              'torch.nn.modules.activation.ReLU'>, stride=1,
>                              groups=1, reduction=None, nh1=None, nh2=None,
>                              dw=False, g2=1, sa=False, sym=False,
>                              norm_type=<NormType.Batch: 1>, act_cls=<class
>                              'torch.nn.modules.activation.ReLU'>, ndim=2,
>                              ks=3, pool=<function AvgPool>, pool_first=True,
>                              padding=None, bias=None, bn_1st=True,
>                              transpose=False, init='auto', xtra=None,
>                              bias_std=0.01,
>                              dilation:Union[int,Tuple[int,int]]=1,
>                              padding_mode:str='zeros', device=None,
>                              dtype=None)

| **Type** | **Default** | **Details**  
---|---|---|---  
c_in |  |  |   
c_out |  |  |   
seq_len | NoneType | None |   
act | type | ReLU |   
stride | int | 1 |   
groups | int | 1 |   
reduction | NoneType | None |   
nh1 | NoneType | None |   
nh2 | NoneType | None |   
dw | bool | False |   
g2 | int | 1 |   
sa | bool | False |   
sym | bool | False |   
norm_type | NormType | NormType.Batch |   
act_cls | type | ReLU |   
ndim | int | 2 |   
ks | int | 3 |   
pool | function | AvgPool |   
pool_first | bool | True |   
padding | NoneType | None |   
bias | NoneType | None |   
bn_1st | bool | True |   
transpose | bool | False |   
init | str | auto |   
xtra | NoneType | None |   
bias_std | float | 0.01 |   
dilation | Union | 1 |   
padding_mode | str | zeros | TODO: refine this type  
device | NoneType | None |   
dtype | NoneType | None |   
  
* * *

source

### xresnet1d34_deeperplus

> 
>      xresnet1d34_deeperplus (c_in, c_out, seq_len=None, act=<class
>                              'torch.nn.modules.activation.ReLU'>, stride=1,
>                              groups=1, reduction=None, nh1=None, nh2=None,
>                              dw=False, g2=1, sa=False, sym=False,
>                              norm_type=<NormType.Batch: 1>, act_cls=<class
>                              'torch.nn.modules.activation.ReLU'>, ndim=2,
>                              ks=3, pool=<function AvgPool>, pool_first=True,
>                              padding=None, bias=None, bn_1st=True,
>                              transpose=False, init='auto', xtra=None,
>                              bias_std=0.01,
>                              dilation:Union[int,Tuple[int,int]]=1,
>                              padding_mode:str='zeros', device=None,
>                              dtype=None)

| **Type** | **Default** | **Details**  
---|---|---|---  
c_in |  |  |   
c_out |  |  |   
seq_len | NoneType | None |   
act | type | ReLU |   
stride | int | 1 |   
groups | int | 1 |   
reduction | NoneType | None |   
nh1 | NoneType | None |   
nh2 | NoneType | None |   
dw | bool | False |   
g2 | int | 1 |   
sa | bool | False |   
sym | bool | False |   
norm_type | NormType | NormType.Batch |   
act_cls | type | ReLU |   
ndim | int | 2 |   
ks | int | 3 |   
pool | function | AvgPool |   
pool_first | bool | True |   
padding | NoneType | None |   
bias | NoneType | None |   
bn_1st | bool | True |   
transpose | bool | False |   
init | str | auto |   
xtra | NoneType | None |   
bias_std | float | 0.01 |   
dilation | Union | 1 |   
padding_mode | str | zeros | TODO: refine this type  
device | NoneType | None |   
dtype | NoneType | None |   
  
* * *

source

### xresnet1d18_deeperplus

> 
>      xresnet1d18_deeperplus (c_in, c_out, seq_len=None, act=<class
>                              'torch.nn.modules.activation.ReLU'>, stride=1,
>                              groups=1, reduction=None, nh1=None, nh2=None,
>                              dw=False, g2=1, sa=False, sym=False,
>                              norm_type=<NormType.Batch: 1>, act_cls=<class
>                              'torch.nn.modules.activation.ReLU'>, ndim=2,
>                              ks=3, pool=<function AvgPool>, pool_first=True,
>                              padding=None, bias=None, bn_1st=True,
>                              transpose=False, init='auto', xtra=None,
>                              bias_std=0.01,
>                              dilation:Union[int,Tuple[int,int]]=1,
>                              padding_mode:str='zeros', device=None,
>                              dtype=None)

| **Type** | **Default** | **Details**  
---|---|---|---  
c_in |  |  |   
c_out |  |  |   
seq_len | NoneType | None |   
act | type | ReLU |   
stride | int | 1 |   
groups | int | 1 |   
reduction | NoneType | None |   
nh1 | NoneType | None |   
nh2 | NoneType | None |   
dw | bool | False |   
g2 | int | 1 |   
sa | bool | False |   
sym | bool | False |   
norm_type | NormType | NormType.Batch |   
act_cls | type | ReLU |   
ndim | int | 2 |   
ks | int | 3 |   
pool | function | AvgPool |   
pool_first | bool | True |   
padding | NoneType | None |   
bias | NoneType | None |   
bn_1st | bool | True |   
transpose | bool | False |   
init | str | auto |   
xtra | NoneType | None |   
bias_std | float | 0.01 |   
dilation | Union | 1 |   
padding_mode | str | zeros | TODO: refine this type  
device | NoneType | None |   
dtype | NoneType | None |   
  
* * *

source

### xresnet1d50_deepplus

> 
>      xresnet1d50_deepplus (c_in, c_out, seq_len=None, act=<class
>                            'torch.nn.modules.activation.ReLU'>, stride=1,
>                            groups=1, reduction=None, nh1=None, nh2=None,
>                            dw=False, g2=1, sa=False, sym=False,
>                            norm_type=<NormType.Batch: 1>, act_cls=<class
>                            'torch.nn.modules.activation.ReLU'>, ndim=2, ks=3,
>                            pool=<function AvgPool>, pool_first=True,
>                            padding=None, bias=None, bn_1st=True,
>                            transpose=False, init='auto', xtra=None,
>                            bias_std=0.01,
>                            dilation:Union[int,Tuple[int,int]]=1,
>                            padding_mode:str='zeros', device=None, dtype=None)

| **Type** | **Default** | **Details**  
---|---|---|---  
c_in |  |  |   
c_out |  |  |   
seq_len | NoneType | None |   
act | type | ReLU |   
stride | int | 1 |   
groups | int | 1 |   
reduction | NoneType | None |   
nh1 | NoneType | None |   
nh2 | NoneType | None |   
dw | bool | False |   
g2 | int | 1 |   
sa | bool | False |   
sym | bool | False |   
norm_type | NormType | NormType.Batch |   
act_cls | type | ReLU |   
ndim | int | 2 |   
ks | int | 3 |   
pool | function | AvgPool |   
pool_first | bool | True |   
padding | NoneType | None |   
bias | NoneType | None |   
bn_1st | bool | True |   
transpose | bool | False |   
init | str | auto |   
xtra | NoneType | None |   
bias_std | float | 0.01 |   
dilation | Union | 1 |   
padding_mode | str | zeros | TODO: refine this type  
device | NoneType | None |   
dtype | NoneType | None |   
  
* * *

source

### xresnet1d34_deepplus

> 
>      xresnet1d34_deepplus (c_in, c_out, seq_len=None, act=<class
>                            'torch.nn.modules.activation.ReLU'>, stride=1,
>                            groups=1, reduction=None, nh1=None, nh2=None,
>                            dw=False, g2=1, sa=False, sym=False,
>                            norm_type=<NormType.Batch: 1>, act_cls=<class
>                            'torch.nn.modules.activation.ReLU'>, ndim=2, ks=3,
>                            pool=<function AvgPool>, pool_first=True,
>                            padding=None, bias=None, bn_1st=True,
>                            transpose=False, init='auto', xtra=None,
>                            bias_std=0.01,
>                            dilation:Union[int,Tuple[int,int]]=1,
>                            padding_mode:str='zeros', device=None, dtype=None)

| **Type** | **Default** | **Details**  
---|---|---|---  
c_in |  |  |   
c_out |  |  |   
seq_len | NoneType | None |   
act | type | ReLU |   
stride | int | 1 |   
groups | int | 1 |   
reduction | NoneType | None |   
nh1 | NoneType | None |   
nh2 | NoneType | None |   
dw | bool | False |   
g2 | int | 1 |   
sa | bool | False |   
sym | bool | False |   
norm_type | NormType | NormType.Batch |   
act_cls | type | ReLU |   
ndim | int | 2 |   
ks | int | 3 |   
pool | function | AvgPool |   
pool_first | bool | True |   
padding | NoneType | None |   
bias | NoneType | None |   
bn_1st | bool | True |   
transpose | bool | False |   
init | str | auto |   
xtra | NoneType | None |   
bias_std | float | 0.01 |   
dilation | Union | 1 |   
padding_mode | str | zeros | TODO: refine this type  
device | NoneType | None |   
dtype | NoneType | None |   
  
* * *

source

### xresnet1d18_deepplus

> 
>      xresnet1d18_deepplus (c_in, c_out, seq_len=None, act=<class
>                            'torch.nn.modules.activation.ReLU'>, stride=1,
>                            groups=1, reduction=None, nh1=None, nh2=None,
>                            dw=False, g2=1, sa=False, sym=False,
>                            norm_type=<NormType.Batch: 1>, act_cls=<class
>                            'torch.nn.modules.activation.ReLU'>, ndim=2, ks=3,
>                            pool=<function AvgPool>, pool_first=True,
>                            padding=None, bias=None, bn_1st=True,
>                            transpose=False, init='auto', xtra=None,
>                            bias_std=0.01,
>                            dilation:Union[int,Tuple[int,int]]=1,
>                            padding_mode:str='zeros', device=None, dtype=None)

| **Type** | **Default** | **Details**  
---|---|---|---  
c_in |  |  |   
c_out |  |  |   
seq_len | NoneType | None |   
act | type | ReLU |   
stride | int | 1 |   
groups | int | 1 |   
reduction | NoneType | None |   
nh1 | NoneType | None |   
nh2 | NoneType | None |   
dw | bool | False |   
g2 | int | 1 |   
sa | bool | False |   
sym | bool | False |   
norm_type | NormType | NormType.Batch |   
act_cls | type | ReLU |   
ndim | int | 2 |   
ks | int | 3 |   
pool | function | AvgPool |   
pool_first | bool | True |   
padding | NoneType | None |   
bias | NoneType | None |   
bn_1st | bool | True |   
transpose | bool | False |   
init | str | auto |   
xtra | NoneType | None |   
bias_std | float | 0.01 |   
dilation | Union | 1 |   
padding_mode | str | zeros | TODO: refine this type  
device | NoneType | None |   
dtype | NoneType | None |   
  
* * *

source

### xresnet1d152plus

> 
>      xresnet1d152plus (c_in, c_out, seq_len=None, act=<class
>                        'torch.nn.modules.activation.ReLU'>, stride=1,
>                        groups=1, reduction=None, nh1=None, nh2=None, dw=False,
>                        g2=1, sa=False, sym=False, norm_type=<NormType.Batch:
>                        1>, act_cls=<class 'torch.nn.modules.activation.ReLU'>,
>                        ndim=2, ks=3, pool=<function AvgPool>, pool_first=True,
>                        padding=None, bias=None, bn_1st=True, transpose=False,
>                        init='auto', xtra=None, bias_std=0.01,
>                        dilation:Union[int,Tuple[int,int]]=1,
>                        padding_mode:str='zeros', device=None, dtype=None)

| **Type** | **Default** | **Details**  
---|---|---|---  
c_in |  |  |   
c_out |  |  |   
seq_len | NoneType | None |   
act | type | ReLU |   
stride | int | 1 |   
groups | int | 1 |   
reduction | NoneType | None |   
nh1 | NoneType | None |   
nh2 | NoneType | None |   
dw | bool | False |   
g2 | int | 1 |   
sa | bool | False |   
sym | bool | False |   
norm_type | NormType | NormType.Batch |   
act_cls | type | ReLU |   
ndim | int | 2 |   
ks | int | 3 |   
pool | function | AvgPool |   
pool_first | bool | True |   
padding | NoneType | None |   
bias | NoneType | None |   
bn_1st | bool | True |   
transpose | bool | False |   
init | str | auto |   
xtra | NoneType | None |   
bias_std | float | 0.01 |   
dilation | Union | 1 |   
padding_mode | str | zeros | TODO: refine this type  
device | NoneType | None |   
dtype | NoneType | None |   
  
* * *

source

### xresnet1d101plus

> 
>      xresnet1d101plus (c_in, c_out, seq_len=None, act=<class
>                        'torch.nn.modules.activation.ReLU'>, stride=1,
>                        groups=1, reduction=None, nh1=None, nh2=None, dw=False,
>                        g2=1, sa=False, sym=False, norm_type=<NormType.Batch:
>                        1>, act_cls=<class 'torch.nn.modules.activation.ReLU'>,
>                        ndim=2, ks=3, pool=<function AvgPool>, pool_first=True,
>                        padding=None, bias=None, bn_1st=True, transpose=False,
>                        init='auto', xtra=None, bias_std=0.01,
>                        dilation:Union[int,Tuple[int,int]]=1,
>                        padding_mode:str='zeros', device=None, dtype=None)

| **Type** | **Default** | **Details**  
---|---|---|---  
c_in |  |  |   
c_out |  |  |   
seq_len | NoneType | None |   
act | type | ReLU |   
stride | int | 1 |   
groups | int | 1 |   
reduction | NoneType | None |   
nh1 | NoneType | None |   
nh2 | NoneType | None |   
dw | bool | False |   
g2 | int | 1 |   
sa | bool | False |   
sym | bool | False |   
norm_type | NormType | NormType.Batch |   
act_cls | type | ReLU |   
ndim | int | 2 |   
ks | int | 3 |   
pool | function | AvgPool |   
pool_first | bool | True |   
padding | NoneType | None |   
bias | NoneType | None |   
bn_1st | bool | True |   
transpose | bool | False |   
init | str | auto |   
xtra | NoneType | None |   
bias_std | float | 0.01 |   
dilation | Union | 1 |   
padding_mode | str | zeros | TODO: refine this type  
device | NoneType | None |   
dtype | NoneType | None |   
  
* * *

source

### xresnet1d50plus

> 
>      xresnet1d50plus (c_in, c_out, seq_len=None, act=<class
>                       'torch.nn.modules.activation.ReLU'>, stride=1, groups=1,
>                       reduction=None, nh1=None, nh2=None, dw=False, g2=1,
>                       sa=False, sym=False, norm_type=<NormType.Batch: 1>,
>                       act_cls=<class 'torch.nn.modules.activation.ReLU'>,
>                       ndim=2, ks=3, pool=<function AvgPool>, pool_first=True,
>                       padding=None, bias=None, bn_1st=True, transpose=False,
>                       init='auto', xtra=None, bias_std=0.01,
>                       dilation:Union[int,Tuple[int,int]]=1,
>                       padding_mode:str='zeros', device=None, dtype=None)

| **Type** | **Default** | **Details**  
---|---|---|---  
c_in |  |  |   
c_out |  |  |   
seq_len | NoneType | None |   
act | type | ReLU |   
stride | int | 1 |   
groups | int | 1 |   
reduction | NoneType | None |   
nh1 | NoneType | None |   
nh2 | NoneType | None |   
dw | bool | False |   
g2 | int | 1 |   
sa | bool | False |   
sym | bool | False |   
norm_type | NormType | NormType.Batch |   
act_cls | type | ReLU |   
ndim | int | 2 |   
ks | int | 3 |   
pool | function | AvgPool |   
pool_first | bool | True |   
padding | NoneType | None |   
bias | NoneType | None |   
bn_1st | bool | True |   
transpose | bool | False |   
init | str | auto |   
xtra | NoneType | None |   
bias_std | float | 0.01 |   
dilation | Union | 1 |   
padding_mode | str | zeros | TODO: refine this type  
device | NoneType | None |   
dtype | NoneType | None |   
  
* * *

source

### xresnet1d34plus

> 
>      xresnet1d34plus (c_in, c_out, seq_len=None, act=<class
>                       'torch.nn.modules.activation.ReLU'>, stride=1, groups=1,
>                       reduction=None, nh1=None, nh2=None, dw=False, g2=1,
>                       sa=False, sym=False, norm_type=<NormType.Batch: 1>,
>                       act_cls=<class 'torch.nn.modules.activation.ReLU'>,
>                       ndim=2, ks=3, pool=<function AvgPool>, pool_first=True,
>                       padding=None, bias=None, bn_1st=True, transpose=False,
>                       init='auto', xtra=None, bias_std=0.01,
>                       dilation:Union[int,Tuple[int,int]]=1,
>                       padding_mode:str='zeros', device=None, dtype=None)

| **Type** | **Default** | **Details**  
---|---|---|---  
c_in |  |  |   
c_out |  |  |   
seq_len | NoneType | None |   
act | type | ReLU |   
stride | int | 1 |   
groups | int | 1 |   
reduction | NoneType | None |   
nh1 | NoneType | None |   
nh2 | NoneType | None |   
dw | bool | False |   
g2 | int | 1 |   
sa | bool | False |   
sym | bool | False |   
norm_type | NormType | NormType.Batch |   
act_cls | type | ReLU |   
ndim | int | 2 |   
ks | int | 3 |   
pool | function | AvgPool |   
pool_first | bool | True |   
padding | NoneType | None |   
bias | NoneType | None |   
bn_1st | bool | True |   
transpose | bool | False |   
init | str | auto |   
xtra | NoneType | None |   
bias_std | float | 0.01 |   
dilation | Union | 1 |   
padding_mode | str | zeros | TODO: refine this type  
device | NoneType | None |   
dtype | NoneType | None |   
  
* * *

source

### xresnet1d18plus

> 
>      xresnet1d18plus (c_in, c_out, seq_len=None, act=<class
>                       'torch.nn.modules.activation.ReLU'>, stride=1, groups=1,
>                       reduction=None, nh1=None, nh2=None, dw=False, g2=1,
>                       sa=False, sym=False, norm_type=<NormType.Batch: 1>,
>                       act_cls=<class 'torch.nn.modules.activation.ReLU'>,
>                       ndim=2, ks=3, pool=<function AvgPool>, pool_first=True,
>                       padding=None, bias=None, bn_1st=True, transpose=False,
>                       init='auto', xtra=None, bias_std=0.01,
>                       dilation:Union[int,Tuple[int,int]]=1,
>                       padding_mode:str='zeros', device=None, dtype=None)

| **Type** | **Default** | **Details**  
---|---|---|---  
c_in |  |  |   
c_out |  |  |   
seq_len | NoneType | None |   
act | type | ReLU |   
stride | int | 1 |   
groups | int | 1 |   
reduction | NoneType | None |   
nh1 | NoneType | None |   
nh2 | NoneType | None |   
dw | bool | False |   
g2 | int | 1 |   
sa | bool | False |   
sym | bool | False |   
norm_type | NormType | NormType.Batch |   
act_cls | type | ReLU |   
ndim | int | 2 |   
ks | int | 3 |   
pool | function | AvgPool |   
pool_first | bool | True |   
padding | NoneType | None |   
bias | NoneType | None |   
bn_1st | bool | True |   
transpose | bool | False |   
init | str | auto |   
xtra | NoneType | None |   
bias_std | float | 0.01 |   
dilation | Union | 1 |   
padding_mode | str | zeros | TODO: refine this type  
device | NoneType | None |   
dtype | NoneType | None |   
      
    
    net = xresnet1d18plus(3, 2, coord=True)
    x = torch.rand(32, 3, 50)
    net(x)__
    
    
    block <class 'tsai.models.layers.ResBlock1dPlus'> expansion 1 layers [2, 2, 2, 2]
    
    
    TensorBase([[ 0.1829,  0.3597],
                [ 0.0274, -0.1443],
                [ 0.0240, -0.2374],
                [-0.1323, -0.6574],
                [ 0.1481, -0.1438],
                [ 0.2410, -0.1225],
                [-0.1186, -0.1978],
                [-0.0640, -0.4547],
                [-0.0229, -0.3214],
                [ 0.2336, -0.4466],
                [-0.1843, -0.0934],
                [-0.0416,  0.1997],
                [-0.0109, -0.0253],
                [ 0.3014, -0.2193],
                [ 0.0966,  0.0602],
                [ 0.2364,  0.2209],
                [-0.1437, -0.1476],
                [ 0.0070, -0.2900],
                [ 0.2807,  0.4797],
                [-0.2386, -0.1563],
                [ 0.1620, -0.2285],
                [ 0.0479, -0.2348],
                [ 0.1573, -0.4420],
                [-0.5469,  0.1512],
                [ 0.0243, -0.1806],
                [ 0.3396,  0.1434],
                [ 0.0666, -0.1644],
                [ 0.3286, -0.5637],
                [ 0.0993, -0.6281],
                [-0.1068, -0.0763],
                [-0.2713,  0.1946],
                [-0.1416, -0.4043]], grad_fn=<AliasBackward0>)
    
    
    bs, c_in, seq_len = 2, 4, 32
    c_out = 2
    x = torch.rand(bs, c_in, seq_len)
    archs = [
        xresnet1d18plus, xresnet1d34plus, xresnet1d50plus, 
        xresnet1d18_deepplus, xresnet1d34_deepplus, xresnet1d50_deepplus, xresnet1d18_deeperplus,
        xresnet1d34_deeperplus, xresnet1d50_deeperplus
    #     # Long test
    #     xresnet1d101, xresnet1d152,
    ]
    for i, arch in enumerate(archs):
        print(i, arch.__name__)
        test_eq(arch(c_in, c_out, sa=True, act=Mish, coord=True)(x).shape, (bs, c_out))__
    
    
    0 xresnet1d18plus
    block <class 'tsai.models.layers.ResBlock1dPlus'> expansion 1 layers [2, 2, 2, 2]
    1 xresnet1d34plus
    block <class 'tsai.models.layers.ResBlock1dPlus'> expansion 1 layers [3, 4, 6, 3]
    2 xresnet1d50plus
    block <class 'tsai.models.layers.ResBlock1dPlus'> expansion 4 layers [3, 4, 6, 3]
    3 xresnet1d18_deepplus
    block <class 'tsai.models.layers.ResBlock1dPlus'> expansion 1 layers [2, 2, 2, 2, 1, 1]
    4 xresnet1d34_deepplus
    block <class 'tsai.models.layers.ResBlock1dPlus'> expansion 1 layers [3, 4, 6, 3, 1, 1]
    5 xresnet1d50_deepplus
    block <class 'tsai.models.layers.ResBlock1dPlus'> expansion 4 layers [3, 4, 6, 3, 1, 1]
    6 xresnet1d18_deeperplus
    block <class 'tsai.models.layers.ResBlock1dPlus'> expansion 1 layers [2, 2, 1, 1, 1, 1, 1, 1]
    7 xresnet1d34_deeperplus
    block <class 'tsai.models.layers.ResBlock1dPlus'> expansion 1 layers [3, 4, 6, 3, 1, 1, 1, 1]
    8 xresnet1d50_deeperplus
    block <class 'tsai.models.layers.ResBlock1dPlus'> expansion 4 layers [3, 4, 6, 3, 1, 1, 1, 1]
    
    
    m = xresnet1d34plus(4, 2, act=Mish)
    test_eq(len(get_layers(m, is_bn)), 38)
    test_eq(check_weight(m, is_bn)[0].sum(), 22)__
    
    
    block <class 'tsai.models.layers.ResBlock1dPlus'> expansion 1 layers [3, 4, 6, 3]

  * __Report an issue


