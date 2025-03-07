## On this page

  * tAPE
  * AbsolutePositionalEncoding
  * LearnablePositionalEncoding
  * Attention
  * Attention_Rel_Scl
  * Attention_Rel_Vec
  * ConvTranBackbone
  * ConvTranPlus



  * __Report an issue



  1. Models
  2. Hybrid models
  3. ConvTranPlus



# ConvTranPlus

> ConvTran: Improving Position Encoding of Transformers for Multivariate Time Series Classification

This is a Pytorch implementation of ConvTran adapted by Ignacio Oguiza and based on:

Foumani, N. M., Tan, C. W., Webb, G. I., & Salehi, M. (2023). Improving Position Encoding of Transformers for Multivariate Time Series Classification. arXiv preprint arXiv:2305.16642.

Pre-print: https://arxiv.org/abs/2305.16642v1

Original repository: https://github.com/Navidfoumani/ConvTran

* * *

source

### tAPE

> 
>      tAPE (d_model:int, seq_len=1024, dropout:float=0.1, scale_factor=1.0)

_time Absolute Position Encoding_

| **Type** | **Default** | **Details**  
---|---|---|---  
d_model | int |  | the embedding dimension  
seq_len | int | 1024 | the max. length of the incoming sequence  
dropout | float | 0.1 | dropout value  
scale_factor | float | 1.0 |   
      
    
    t = torch.randn(8, 50, 128)
    assert tAPE(128, 50)(t).shape == t.shape __

* * *

source

### AbsolutePositionalEncoding

> 
>      AbsolutePositionalEncoding (d_model:int, seq_len=1024, dropout:float=0.1,
>                                  scale_factor=1.0)

_Absolute positional encoding_

| **Type** | **Default** | **Details**  
---|---|---|---  
d_model | int |  | the embedding dimension  
seq_len | int | 1024 | the max. length of the incoming sequence  
dropout | float | 0.1 | dropout value  
scale_factor | float | 1.0 |   
      
    
    t = torch.randn(8, 50, 128)
    assert AbsolutePositionalEncoding(128, 50)(t).shape == t.shape __

* * *

source

### LearnablePositionalEncoding

> 
>      LearnablePositionalEncoding (d_model:int, seq_len=1024,
>                                   dropout:float=0.1)

_Learnable positional encoding_

| **Type** | **Default** | **Details**  
---|---|---|---  
d_model | int |  | the embedding dimension  
seq_len | int | 1024 | the max. length of the incoming sequence  
dropout | float | 0.1 | dropout value  
      
    
    t = torch.randn(8, 50, 128)
    assert LearnablePositionalEncoding(128, 50)(t).shape == t.shape __

* * *

source

### Attention

> 
>      Attention (d_model:int, n_heads:int=8, dropout:float=0.01)

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
d_model | int |  | Embedding dimension  
n_heads | int | 8 | number of attention heads  
dropout | float | 0.01 | dropout  
      
    
    t = torch.randn(8, 50, 128)
    assert Attention(128)(t).shape == t.shape __

* * *

source

### Attention_Rel_Scl

> 
>      Attention_Rel_Scl (d_model:int, seq_len:int, n_heads:int=8,
>                         dropout:float=0.01)

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
d_model | int |  | Embedding dimension  
seq_len | int |  | sequence length  
n_heads | int | 8 | number of attention heads  
dropout | float | 0.01 | dropout  
      
    
    t = torch.randn(8, 50, 128)
    assert Attention_Rel_Scl(128, 50)(t).shape == t.shape __

* * *

source

### Attention_Rel_Vec

> 
>      Attention_Rel_Vec (d_model:int, seq_len:int, n_heads:int=8,
>                         dropout:float=0.01)

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
d_model | int |  | Embedding dimension  
seq_len | int |  | sequence length  
n_heads | int | 8 | number of attention heads  
dropout | float | 0.01 | dropout  
      
    
    t = torch.randn(8, 50, 128)
    assert Attention_Rel_Vec(128, 50)(t).shape == t.shape __

* * *

source

### ConvTranBackbone

> 
>      ConvTranBackbone (c_in:int, seq_len:int, d_model=16, n_heads:int=8,
>                        dim_ff:int=256, abs_pos_encode:str='tAPE',
>                        rel_pos_encode:str='eRPE', dropout:float=0.01)

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
c_in | int |  |   
seq_len | int |  |   
d_model | int | 16 | Internal dimension of transformer embeddings  
n_heads | int | 8 | Number of multi-headed attention heads  
dim_ff | int | 256 | Dimension of dense feedforward part of transformer layer  
abs_pos_encode | str | tAPE | Absolute Position Embedding. choices={‘tAPE’, ‘sin’, ‘learned’, None}  
rel_pos_encode | str | eRPE | Relative Position Embedding. choices={‘eRPE’, ‘vector’, None}  
dropout | float | 0.01 | Droupout regularization ratio  
      
    
    t = torch.randn(8, 5, 20)
    assert ConvTranBackbone(5, 20)(t).shape, (8, 16, 20)__

* * *

source

### ConvTranPlus

> 
>      ConvTranPlus (c_in:int, c_out:int, seq_len:int, d:tuple=None,
>                    d_model:int=16, n_heads:int=8, dim_ff:int=256,
>                    abs_pos_encode:str='tAPE', rel_pos_encode:str='eRPE',
>                    encoder_dropout:float=0.01, fc_dropout:float=0.1,
>                    use_bn:bool=True, flatten:bool=True, custom_head:Any=None)

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
c_in | int |  | Number of channels in input  
c_out | int |  | Number of channels in output  
seq_len | int |  | Number of input sequence length  
d | tuple | None | output shape (excluding batch dimension).  
d_model | int | 16 | Internal dimension of transformer embeddings  
n_heads | int | 8 | Number of multi-headed attention heads  
dim_ff | int | 256 | Dimension of dense feedforward part of transformer layer  
abs_pos_encode | str | tAPE | Absolute Position Embedding. choices={‘tAPE’, ‘sin’, ‘learned’, None}  
rel_pos_encode | str | eRPE | Relative Position Embedding. choices={‘eRPE’, ‘vector’, None}  
encoder_dropout | float | 0.01 | Droupout regularization ratio for the encoder  
fc_dropout | float | 0.1 | Droupout regularization ratio for the head  
use_bn | bool | True | indicates if batchnorm will be applied to the model head.  
flatten | bool | True | this will flatten the output of the encoder before applying the head if True.  
custom_head | Any | None | custom head that will be applied to the model head (optional).  
      
    
    xb = torch.randn(16, 5, 20)
    
    model = ConvTranPlus(5, 3, 20, d=None)
    output = model(xb)
    assert output.shape == (16, 3)__
    
    
    xb = torch.randn(16, 5, 20)
    
    model = ConvTranPlus(5, 3, 20, d=5)
    output = model(xb)
    assert output.shape == (16, 5, 3)__
    
    
    xb = torch.randn(16, 5, 20)
    
    model = ConvTranPlus(5, 3, 20, d=(2, 10))
    output = model(xb)
    assert output.shape == (16, 2, 10, 3)__

  * __Report an issue


