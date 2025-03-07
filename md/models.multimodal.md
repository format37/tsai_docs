## On this page

  * get_feat_idxs
  * get_o_cont_idxs
  * TensorSplitter
  * Embeddings
  * StaticBackbone
  * FusionMLP
  * MultInputBackboneWrapper
  * MultInputWrapper



  * __Report an issue



  1. Models
  2. Miscellaneous
  3. Multimodal



# Multimodal

> Functionality used for multiple data modalities.

A common scenario in time-series related tasks is the use of multiple types of inputs:

  * static: data that doesn’t change with time
  * observed: temporal data only available in the past
  * known: temporal data available in the past and in the future



At the same time, these different modalities may contain:

  * categorical data
  * continuous or numerical data



Based on that, there are situations where we have up to 6 different types of input features:

  * s_cat: static continuous variables
  * o_cat: observed categorical variables
  * o_cont: observed continuous variables
  * k_cat: known categorical variables
  * k_cont: known continuous variables



* * *

source

### get_feat_idxs

> 
>      get_feat_idxs (c_in, s_cat_idxs=None, s_cont_idxs=None, o_cat_idxs=None,
>                     o_cont_idxs=None)

_Calculate the indices of the features used for training._

* * *

source

### get_o_cont_idxs

> 
>      get_o_cont_idxs (c_in, s_cat_idxs=None, s_cont_idxs=None,
>                       o_cat_idxs=None)

_Calculate the indices of the observed continuous features._
    
    
    c_in = 7
    s_cat_idxs = 3
    s_cont_idxs = [1, 4, 5]
    o_cat_idxs = None
    o_cont_idxs = None
    
    s_cat_idxs, s_cont_idxs, o_cat_idxs, o_cont_idxs = get_feat_idxs(c_in, s_cat_idxs=s_cat_idxs, s_cont_idxs=s_cont_idxs, o_cat_idxs=o_cat_idxs, o_cont_idxs=o_cont_idxs)
    
    test_eq(s_cat_idxs, [3])
    test_eq(s_cont_idxs, [1, 4, 5])
    test_eq(o_cat_idxs, [])
    test_eq(o_cont_idxs, [0, 2, 6])__

* * *

source

### TensorSplitter

> 
>      TensorSplitter (s_cat_idxs:list=None, s_cont_idxs:list=None,
>                      o_cat_idxs:list=None, o_cont_idxs:list=None,
>                      k_cat_idxs:list=None, k_cont_idxs:list=None,
>                      horizon:int=None)

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

ivar training: Boolean represents whether this module is in training or evaluation mode. :vartype training: bool* | **Type** | **Default** | **Details**  
---|---|---|---  
s_cat_idxs | list | None | list of indices for static categorical variables  
s_cont_idxs | list | None | list of indices for static continuous variables  
o_cat_idxs | list | None | list of indices for observed categorical variables  
o_cont_idxs | list | None | list of indices for observed continuous variables  
k_cat_idxs | list | None | list of indices for known categorical variables  
k_cont_idxs | list | None | list of indices for known continuous variables  
horizon | int | None | number of time steps to predict ahead  
      
    
    # Example usage
    bs = 4
    s_cat_idxs = 1
    s_cont_idxs = [0, 2]
    o_cat_idxs =[ 3, 4, 5]
    o_cont_idxs = None
    k_cat_idxs = None
    k_cont_idxs = None
    horizon=None
    input_tensor = torch.randn(bs, 6, 10)  # 3D input tensor
    splitter = TensorSplitter(s_cat_idxs=s_cat_idxs, s_cont_idxs=s_cont_idxs,
                              o_cat_idxs=o_cat_idxs, o_cont_idxs=o_cont_idxs)
    slices = splitter(input_tensor)
    for i, slice_tensor in enumerate(slices):
        print(f"Slice {i+1}: {slice_tensor.shape} {slice_tensor.dtype}")__
    
    
    Slice 1: torch.Size([4, 1]) torch.int64
    Slice 2: torch.Size([4, 2]) torch.int64
    Slice 3: torch.Size([4, 3, 10]) torch.float32
    Slice 4: torch.Size([4, 0, 10]) torch.float32
    
    
    # Example usage
    bs = 4
    s_cat_idxs = 1
    s_cont_idxs = [0, 2]
    o_cat_idxs =[ 3, 4, 5]
    o_cont_idxs = None
    k_cat_idxs = [6,7]
    k_cont_idxs = 8
    horizon=3
    input_tensor = torch.randn(4, 9, 10)  # 3D input tensor
    splitter = TensorSplitter(s_cat_idxs=s_cat_idxs, s_cont_idxs=s_cont_idxs,
                              o_cat_idxs=o_cat_idxs, o_cont_idxs=o_cont_idxs,
                              k_cat_idxs=k_cat_idxs, k_cont_idxs=k_cont_idxs, horizon=horizon)
    slices = splitter(input_tensor)
    for i, slice_tensor in enumerate(slices):
        print(f"Slice {i+1}: {slice_tensor.shape} {slice_tensor.dtype}")__
    
    
    Slice 1: torch.Size([4, 1]) torch.int64
    Slice 2: torch.Size([4, 2]) torch.int64
    Slice 3: torch.Size([4, 3, 7]) torch.float32
    Slice 4: torch.Size([4, 0, 7]) torch.float32
    Slice 5: torch.Size([4, 2, 10]) torch.float32
    Slice 6: torch.Size([4, 1, 10]) torch.float32

* * *

source

### Embeddings

> 
>      Embeddings (n_embeddings:list, embedding_dims:list=None,
>                  padding_idx:int=0, embed_dropout:float=0.0, **kwargs)

_Embedding layers for each categorical variable in a 2D or 3D tensor_

| **Type** | **Default** | **Details**  
---|---|---|---  
n_embeddings | list |  | List of num_embeddings for each categorical variable  
embedding_dims | list | None | List of embedding dimensions for each categorical variable  
padding_idx | int | 0 | Embedding padding_idx  
embed_dropout | float | 0.0 | Dropout probability for `Embedding` layer  
kwargs | VAR_KEYWORD |  |   
      
    
    t1 = torch.randint(0, 7, (16, 1))
    t2 = torch.randint(0, 5, (16, 1))
    t = torch.cat([t1, t2], 1).float()
    emb = Embeddings([7, 5], None, embed_dropout=0.1)
    test_eq(emb(t).shape, (16, 12))__
    
    
    t1 = torch.randint(0, 7, (16, 1))
    t2 = torch.randint(0, 5, (16, 1))
    t = torch.cat([t1, t2], 1).float()
    emb = Embeddings([7, 5], [4, 3])
    test_eq(emb(t).shape, (16, 12))__
    
    
    t1 = torch.randint(0, 7, (16, 1, 10))
    t2 = torch.randint(0, 5, (16, 1, 10))
    t = torch.cat([t1, t2], 1).float()
    emb = Embeddings([7, 5], None)
    test_eq(emb(t).shape, (16, 12, 10))__

* * *

source

### StaticBackbone

> 
>      StaticBackbone (c_in, c_out, seq_len, d=None, layers=[200, 100],
>                      dropouts=[0.1, 0.2], act=ReLU(inplace=True),
>                      use_bn=False, lin_first=False)

_Static backbone model to embed static features_
    
    
    # Example usage
    bs = 4
    c_in = 6
    c_out = 8
    seq_len = 10
    input_tensor = torch.randn(bs, c_in, seq_len)  # 3D input tensor
    backbone = StaticBackbone(c_in, c_out, seq_len)
    output_tensor = backbone(input_tensor)
    print(f"Input shape: {input_tensor.shape} Output shape: {output_tensor.shape}")
    backbone __
    
    
    Input shape: torch.Size([4, 6, 10]) Output shape: torch.Size([4, 100])
    
    
    StaticBackbone(
      (flatten): Reshape(bs)
      (mlp): ModuleList(
        (0): LinBnDrop(
          (0): Dropout(p=0.1, inplace=False)
          (1): Linear(in_features=60, out_features=200, bias=True)
          (2): ReLU(inplace=True)
        )
        (1): LinBnDrop(
          (0): Dropout(p=0.2, inplace=False)
          (1): Linear(in_features=200, out_features=100, bias=True)
          (2): ReLU(inplace=True)
        )
      )
    )

* * *

source

### FusionMLP

> 
>      FusionMLP (comb_dim, layers, act='relu', dropout=0.0, use_bn=True)

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
    
    
    bs = 16
    emb_dim = 128
    seq_len = 20
    cat_dim = 24
    cont_feat = 3
    
    comb_dim = emb_dim + cat_dim + cont_feat
    emb = torch.randn(bs, emb_dim, seq_len)
    cat = torch.randn(bs, cat_dim)
    cont = torch.randn(bs, cont_feat)
    fusion_mlp = FusionMLP(comb_dim, layers=comb_dim, act='relu', dropout=.1)
    output = fusion_mlp(cat, cont, emb)
    test_eq(output.shape, (bs, comb_dim))__
    
    
    bs = 16
    emb_dim = 50000
    cat_dim = 24
    cont_feat = 3
    
    comb_dim = emb_dim + cat_dim + cont_feat
    emb = torch.randn(bs, emb_dim)
    cat = torch.randn(bs, cat_dim)
    cont = torch.randn(bs, cont_feat)
    fusion_mlp = FusionMLP(comb_dim, layers=[128], act='relu', dropout=.1)
    output = fusion_mlp(cat, cont, emb)
    test_eq(output.shape, (bs, 128))__

* * *

source

### MultInputBackboneWrapper

> 
>      MultInputBackboneWrapper (arch, c_in:int=None, seq_len:int=None,
>                                d:tuple=None,
>                                dls:tsai.data.core.TSDataLoaders=None,
>                                s_cat_idxs:list=None,
>                                s_cat_embeddings:list=None,
>                                s_cat_embedding_dims:list=None,
>                                s_cont_idxs:list=None, o_cat_idxs:list=None,
>                                o_cat_embeddings:list=None,
>                                o_cat_embedding_dims:list=None,
>                                o_cont_idxs:list=None, patch_len:int=None,
>                                patch_stride:int=None,
>                                fusion_layers:list=[128],
>                                fusion_act:str='relu',
>                                fusion_dropout:float=0.0,
>                                fusion_use_bn:bool=True, **kwargs)

_Model backbone wrapper for input tensors with static and/ or observed, categorical and/ or numerical features._

| **Type** | **Default** | **Details**  
---|---|---|---  
arch |  |  |   
c_in | int | None | number of input variables  
seq_len | int | None | input sequence length  
d | tuple | None | shape of the output tensor  
dls | TSDataLoaders | None | TSDataLoaders object  
s_cat_idxs | list | None | list of indices for static categorical variables  
s_cat_embeddings | list | None | list of num_embeddings for each static categorical variable  
s_cat_embedding_dims | list | None | list of embedding dimensions for each static categorical variable  
s_cont_idxs | list | None | list of indices for static continuous variables  
o_cat_idxs | list | None | list of indices for observed categorical variables  
o_cat_embeddings | list | None | list of num_embeddings for each observed categorical variable  
o_cat_embedding_dims | list | None | list of embedding dimensions for each observed categorical variable  
o_cont_idxs | list | None | list of indices for observed continuous variables. All features not in s_cat_idxs, s_cont_idxs, o_cat_idxs are considered observed continuous variables.  
patch_len | int | None | Number of time steps in each patch.  
patch_stride | int | None | Stride of the patch.  
fusion_layers | list | [128] | list of layer dimensions for the fusion MLP  
fusion_act | str | relu | activation function for the fusion MLP  
fusion_dropout | float | 0.0 | dropout probability for the fusion MLP  
fusion_use_bn | bool | True | boolean indicating whether to use batch normalization in the fusion MLP  
kwargs | VAR_KEYWORD |  |   
  
* * *

source

### MultInputWrapper

> 
>      MultInputWrapper (arch, c_in:int=None, c_out:int=1, seq_len:int=None,
>                        d:tuple=None, dls:tsai.data.core.TSDataLoaders=None,
>                        s_cat_idxs:list=None, s_cat_embeddings:list=None,
>                        s_cat_embedding_dims:list=None, s_cont_idxs:list=None,
>                        o_cat_idxs:list=None, o_cat_embeddings:list=None,
>                        o_cat_embedding_dims:list=None, o_cont_idxs:list=None,
>                        patch_len:int=None, patch_stride:int=None,
>                        fusion_layers:list=128, fusion_act:str='relu',
>                        fusion_dropout:float=0.0, fusion_use_bn:bool=True,
>                        custom_head=None, **kwargs)

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
arch |  |  |   
c_in | int | None | number of input variables  
c_out | int | 1 | number of output variables  
seq_len | int | None | input sequence length  
d | tuple | None | shape of the output tensor  
dls | TSDataLoaders | None | TSDataLoaders object  
s_cat_idxs | list | None | list of indices for static categorical variables  
s_cat_embeddings | list | None | list of num_embeddings for each static categorical variable  
s_cat_embedding_dims | list | None | list of embedding dimensions for each static categorical variable  
s_cont_idxs | list | None | list of indices for static continuous variables  
o_cat_idxs | list | None | list of indices for observed categorical variables  
o_cat_embeddings | list | None | list of num_embeddings for each observed categorical variable  
o_cat_embedding_dims | list | None | list of embedding dimensions for each observed categorical variable  
o_cont_idxs | list | None | list of indices for observed continuous variables. All features not in s_cat_idxs, s_cont_idxs, o_cat_idxs are considered observed continuous variables.  
patch_len | int | None | Number of time steps in each patch.  
patch_stride | int | None | Stride of the patch.  
fusion_layers | list | 128 | list of layer dimensions for the fusion MLP  
fusion_act | str | relu | activation function for the fusion MLP  
fusion_dropout | float | 0.0 | dropout probability for the fusion MLP  
fusion_use_bn | bool | True | boolean indicating whether to use batch normalization in the fusion MLP  
custom_head | NoneType | None | custom head to replace the default head  
kwargs | VAR_KEYWORD |  |   
      
    
    from tsai.models.InceptionTimePlus import InceptionTimePlus __
    
    
    bs = 8
    c_in = 6
    c_out = 3
    seq_len = 97
    d = None
    
    s_cat_idxs=2
    s_cont_idxs=4
    o_cat_idxs=[0, 3]
    o_cont_idxs=None
    s_cat_embeddings = 5
    s_cat_embedding_dims = None
    o_cat_embeddings = [7, 3]
    o_cat_embedding_dims = [3, None]
    
    fusion_layers = 128
    
    t0 = torch.randint(0, 7, (bs, 1, seq_len)) # cat
    t1 = torch.randn(bs, 1, seq_len)
    t2 = torch.randint(0, 5, (bs, 1, seq_len)) # cat
    t3 = torch.randint(0, 3, (bs, 1, seq_len)) # cat
    t4 = torch.randn(bs, 1, seq_len)
    t5 = torch.randn(bs, 1, seq_len)
    
    t = torch.cat([t0, t1, t2, t3, t4, t5], 1).float().to(default_device())
    
    patch_lens = [None, 5, 5, 5, 5]
    patch_strides = [None, None, 1, 3, 5]
    for patch_len, patch_stride in zip(patch_lens, patch_strides):
        for arch in ["InceptionTimePlus", InceptionTimePlus, "TSiTPlus"]:
            print(f"arch: {arch}, patch_len: {patch_len}, patch_stride: {patch_stride}")
    
            model = MultInputWrapper(
                arch=arch,
                c_in=c_in,
                c_out=c_out,
                seq_len=seq_len,
                d=d,
                s_cat_idxs=s_cat_idxs, s_cat_embeddings=s_cat_embeddings, s_cat_embedding_dims=s_cat_embedding_dims,
                s_cont_idxs=s_cont_idxs,
                o_cat_idxs=o_cat_idxs, o_cat_embeddings=o_cat_embeddings, o_cat_embedding_dims=o_cat_embedding_dims,
                o_cont_idxs=o_cont_idxs,
                patch_len=patch_len,
                patch_stride=patch_stride,
                fusion_layers=fusion_layers,
            ).to(default_device())
    
            test_eq(model(t).shape, (bs, c_out))__
    
    
    arch: InceptionTimePlus, patch_len: None, patch_stride: None
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: None, patch_stride: None
    arch: TSiTPlus, patch_len: None, patch_stride: None
    arch: InceptionTimePlus, patch_len: 5, patch_stride: None
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: None
    arch: TSiTPlus, patch_len: 5, patch_stride: None
    arch: InceptionTimePlus, patch_len: 5, patch_stride: 1
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: 1
    arch: TSiTPlus, patch_len: 5, patch_stride: 1
    arch: InceptionTimePlus, patch_len: 5, patch_stride: 3
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: 3
    arch: TSiTPlus, patch_len: 5, patch_stride: 3
    arch: InceptionTimePlus, patch_len: 5, patch_stride: 5
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: 5
    arch: TSiTPlus, patch_len: 5, patch_stride: 5
    
    
    bs = 8
    c_in = 6
    c_out = 3
    seq_len = 97
    d = None
    
    s_cat_idxs=None
    s_cont_idxs=4
    o_cat_idxs=[0, 3]
    o_cont_idxs=None
    s_cat_embeddings = None
    s_cat_embedding_dims = None
    o_cat_embeddings = [7, 3]
    o_cat_embedding_dims = [3, None]
    
    fusion_layers = 128
    
    t0 = torch.randint(0, 7, (bs, 1, seq_len)) # cat
    t1 = torch.randn(bs, 1, seq_len)
    t2 = torch.randint(0, 5, (bs, 1, seq_len)) # cat
    t3 = torch.randint(0, 3, (bs, 1, seq_len)) # cat
    t4 = torch.randn(bs, 1, seq_len)
    t5 = torch.randn(bs, 1, seq_len)
    
    t = torch.cat([t0, t1, t2, t3, t4, t5], 1).float().to(default_device())
    
    patch_lens = [None, 5, 5, 5, 5]
    patch_strides = [None, None, 1, 3, 5]
    for patch_len, patch_stride in zip(patch_lens, patch_strides):
        for arch in ["InceptionTimePlus", InceptionTimePlus, "TSiTPlus"]:
            print(f"arch: {arch}, patch_len: {patch_len}, patch_stride: {patch_stride}")
    
            model = MultInputWrapper(
                arch=arch,
                c_in=c_in,
                c_out=c_out,
                seq_len=seq_len,
                d=d,
                s_cat_idxs=s_cat_idxs, s_cat_embeddings=s_cat_embeddings, s_cat_embedding_dims=s_cat_embedding_dims,
                s_cont_idxs=s_cont_idxs,
                o_cat_idxs=o_cat_idxs, o_cat_embeddings=o_cat_embeddings, o_cat_embedding_dims=o_cat_embedding_dims,
                o_cont_idxs=o_cont_idxs,
                patch_len=patch_len,
                patch_stride=patch_stride,
                fusion_layers=fusion_layers,
            ).to(default_device())
    
            test_eq(model(t).shape, (bs, c_out))__
    
    
    arch: InceptionTimePlus, patch_len: None, patch_stride: None
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: None, patch_stride: None
    arch: TSiTPlus, patch_len: None, patch_stride: None
    arch: InceptionTimePlus, patch_len: 5, patch_stride: None
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: None
    arch: TSiTPlus, patch_len: 5, patch_stride: None
    arch: InceptionTimePlus, patch_len: 5, patch_stride: 1
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: 1
    arch: TSiTPlus, patch_len: 5, patch_stride: 1
    arch: InceptionTimePlus, patch_len: 5, patch_stride: 3
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: 3
    arch: TSiTPlus, patch_len: 5, patch_stride: 3
    arch: InceptionTimePlus, patch_len: 5, patch_stride: 5
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: 5
    arch: TSiTPlus, patch_len: 5, patch_stride: 5
    
    
    bs = 8
    c_in = 6
    c_out = 3
    seq_len = 97
    d = None
    
    s_cat_idxs=2
    s_cont_idxs=4
    o_cat_idxs=None
    o_cont_idxs=None
    s_cat_embeddings = 5
    s_cat_embedding_dims = None
    o_cat_embeddings = None
    o_cat_embedding_dims = None
    
    fusion_layers = 128
    
    t0 = torch.randint(0, 7, (bs, 1, seq_len)) # cat
    t1 = torch.randn(bs, 1, seq_len)
    t2 = torch.randint(0, 5, (bs, 1, seq_len)) # cat
    t3 = torch.randint(0, 3, (bs, 1, seq_len)) # cat
    t4 = torch.randn(bs, 1, seq_len)
    t5 = torch.randn(bs, 1, seq_len)
    
    t = torch.cat([t0, t1, t2, t3, t4, t5], 1).float().to(default_device())
    
    patch_lens = [None, 5, 5, 5, 5]
    patch_strides = [None, None, 1, 3, 5]
    for patch_len, patch_stride in zip(patch_lens, patch_strides):
        for arch in ["InceptionTimePlus", InceptionTimePlus, "TSiTPlus"]:
            print(f"arch: {arch}, patch_len: {patch_len}, patch_stride: {patch_stride}")
    
            model = MultInputWrapper(
                arch=arch,
                c_in=c_in,
                c_out=c_out,
                seq_len=seq_len,
                d=d,
                s_cat_idxs=s_cat_idxs, s_cat_embeddings=s_cat_embeddings, s_cat_embedding_dims=s_cat_embedding_dims,
                s_cont_idxs=s_cont_idxs,
                o_cat_idxs=o_cat_idxs, o_cat_embeddings=o_cat_embeddings, o_cat_embedding_dims=o_cat_embedding_dims,
                o_cont_idxs=o_cont_idxs,
                patch_len=patch_len,
                patch_stride=patch_stride,
                fusion_layers=fusion_layers,
            ).to(default_device())
    
            test_eq(model(t).shape, (bs, c_out))__
    
    
    arch: InceptionTimePlus, patch_len: None, patch_stride: None
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: None, patch_stride: None
    arch: TSiTPlus, patch_len: None, patch_stride: None
    arch: InceptionTimePlus, patch_len: 5, patch_stride: None
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: None
    arch: TSiTPlus, patch_len: 5, patch_stride: None
    arch: InceptionTimePlus, patch_len: 5, patch_stride: 1
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: 1
    arch: TSiTPlus, patch_len: 5, patch_stride: 1
    arch: InceptionTimePlus, patch_len: 5, patch_stride: 3
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: 3
    arch: TSiTPlus, patch_len: 5, patch_stride: 3
    arch: InceptionTimePlus, patch_len: 5, patch_stride: 5
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: 5
    arch: TSiTPlus, patch_len: 5, patch_stride: 5
    
    
    bs = 8
    c_in = 6
    c_out = 3
    seq_len = 97
    d = None
    
    s_cat_idxs=None
    s_cont_idxs=None
    o_cat_idxs=None
    o_cont_idxs=None
    s_cat_embeddings = None
    s_cat_embedding_dims = None
    o_cat_embeddings = None
    o_cat_embedding_dims = None
    
    fusion_layers = 128
    
    t0 = torch.randint(0, 7, (bs, 1, seq_len)) # cat
    t1 = torch.randn(bs, 1, seq_len)
    t2 = torch.randint(0, 5, (bs, 1, seq_len)) # cat
    t3 = torch.randint(0, 3, (bs, 1, seq_len)) # cat
    t4 = torch.randn(bs, 1, seq_len)
    t5 = torch.randn(bs, 1, seq_len)
    
    t = torch.cat([t0, t1, t2, t3, t4, t5], 1).float().to(default_device())
    
    patch_lens = [None, 5, 5, 5, 5]
    patch_strides = [None, None, 1, 3, 5]
    for patch_len, patch_stride in zip(patch_lens, patch_strides):
        for arch in ["InceptionTimePlus", InceptionTimePlus, "TSiTPlus"]:
            print(f"arch: {arch}, patch_len: {patch_len}, patch_stride: {patch_stride}")
    
            model = MultInputWrapper(
                arch=arch,
                c_in=c_in,
                c_out=c_out,
                seq_len=seq_len,
                d=d,
                s_cat_idxs=s_cat_idxs, s_cat_embeddings=s_cat_embeddings, s_cat_embedding_dims=s_cat_embedding_dims,
                s_cont_idxs=s_cont_idxs,
                o_cat_idxs=o_cat_idxs, o_cat_embeddings=o_cat_embeddings, o_cat_embedding_dims=o_cat_embedding_dims,
                o_cont_idxs=o_cont_idxs,
                patch_len=patch_len,
                patch_stride=patch_stride,
                fusion_layers=fusion_layers,
            ).to(default_device())
    
            test_eq(model(t).shape, (bs, c_out))__
    
    
    arch: InceptionTimePlus, patch_len: None, patch_stride: None
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: None, patch_stride: None
    arch: TSiTPlus, patch_len: None, patch_stride: None
    arch: InceptionTimePlus, patch_len: 5, patch_stride: None
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: None
    arch: TSiTPlus, patch_len: 5, patch_stride: None
    arch: InceptionTimePlus, patch_len: 5, patch_stride: 1
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: 1
    arch: TSiTPlus, patch_len: 5, patch_stride: 1
    arch: InceptionTimePlus, patch_len: 5, patch_stride: 3
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: 3
    arch: TSiTPlus, patch_len: 5, patch_stride: 3
    arch: InceptionTimePlus, patch_len: 5, patch_stride: 5
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: 5
    arch: TSiTPlus, patch_len: 5, patch_stride: 5
    
    
    class CustomHead(nn.Module):
        def __init__(self, head_nf, c_out, seq_len, d):
            super().__init__()
            self.d = d
            self.c_out = c_out
            self.fc = nn.Linear(head_nf, d * c_out)
    
        def forward(self, x):
            x = self.fc(x)         # [batch_size, d*c]
            x = x.view(x.shape[0], self.d, self.c_out)
            return x __
    
    
    bs = 8
    c_in = 6
    c_out = 3
    seq_len = 97
    d = 7
    
    s_cat_idxs=None
    s_cont_idxs=None
    o_cat_idxs=None
    o_cont_idxs=None
    s_cat_embeddings = None
    s_cat_embedding_dims = None
    o_cat_embeddings = None
    o_cat_embedding_dims = None
    
    fusion_layers = 128
    
    t0 = torch.randint(0, 7, (bs, 1, seq_len)) # cat
    t1 = torch.randn(bs, 1, seq_len)
    t2 = torch.randint(0, 5, (bs, 1, seq_len)) # cat
    t3 = torch.randint(0, 3, (bs, 1, seq_len)) # cat
    t4 = torch.randn(bs, 1, seq_len)
    t5 = torch.randn(bs, 1, seq_len)
    
    t = torch.cat([t0, t1, t2, t3, t4, t5], 1).float().to(default_device())
    
    patch_lens = [None, 5, 5, 5, 5]
    patch_strides = [None, None, 1, 3, 5]
    for patch_len, patch_stride in zip(patch_lens, patch_strides):
        for arch in ["InceptionTimePlus", InceptionTimePlus, "TSiTPlus"]:
            print(f"arch: {arch}, patch_len: {patch_len}, patch_stride: {patch_stride}")
            model = MultInputWrapper(
                arch=arch,
                custom_head=CustomHead,
                c_in=c_in,
                c_out=c_out,
                seq_len=seq_len,
                d=d,
                s_cat_idxs=s_cat_idxs, s_cat_embeddings=s_cat_embeddings, s_cat_embedding_dims=s_cat_embedding_dims,
                s_cont_idxs=s_cont_idxs,
                o_cat_idxs=o_cat_idxs, o_cat_embeddings=o_cat_embeddings, o_cat_embedding_dims=o_cat_embedding_dims,
                o_cont_idxs=o_cont_idxs,
                patch_len=patch_len,
                patch_stride=patch_stride,
                fusion_layers=fusion_layers,
            ).to(default_device())
    
            test_eq(model(t).shape, (bs, d, c_out))__
    
    
    arch: InceptionTimePlus, patch_len: None, patch_stride: None
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: None, patch_stride: None
    arch: TSiTPlus, patch_len: None, patch_stride: None
    arch: InceptionTimePlus, patch_len: 5, patch_stride: None
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: None
    arch: TSiTPlus, patch_len: 5, patch_stride: None
    arch: InceptionTimePlus, patch_len: 5, patch_stride: 1
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: 1
    arch: TSiTPlus, patch_len: 5, patch_stride: 1
    arch: InceptionTimePlus, patch_len: 5, patch_stride: 3
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: 3
    arch: TSiTPlus, patch_len: 5, patch_stride: 3
    arch: InceptionTimePlus, patch_len: 5, patch_stride: 5
    arch: <class 'tsai.models.InceptionTimePlus.InceptionTimePlus'>, patch_len: 5, patch_stride: 5
    arch: TSiTPlus, patch_len: 5, patch_stride: 5

  * __Report an issue


