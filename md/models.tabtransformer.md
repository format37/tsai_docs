## On this page

  * TabTransformer
  * FullEmbeddingDropout
  * SharedEmbedding
  * ifnone



  * __Report an issue



  1. Models
  2. Tabular models
  3. TabTransformer



# TabTransformer

This is an unofficial TabTransformer Pytorch implementation created by Ignacio Oguiza (oguiza@timeseriesAI.co)

Huang, X., Khetan, A., Cvitkovic, M., & Karnin, Z. (2020). **TabTransformer: Tabular Data Modeling Using Contextual Embeddings**. arXiv preprint https://arxiv.org/pdf/2012.06678

Official repo: https://github.com/awslabs/autogluon/tree/master/tabular/src/autogluon/tabular/models/tab_transformer

* * *

source

### TabTransformer

> 
>      TabTransformer (classes, cont_names, c_out, column_embed=True,
>                      add_shared_embed=False, shared_embed_div=8,
>                      embed_dropout=0.1, drop_whole_embed=False, d_model=32,
>                      n_layers=6, n_heads=8, d_k=None, d_v=None, d_ff=None,
>                      res_attention=True, attention_act='gelu',
>                      res_dropout=0.1, norm_cont=True, mlp_mults=(4, 2),
>                      mlp_dropout=0.0, mlp_act=None, mlp_skip=False,
>                      mlp_bn=False, bn_final=False)

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

### FullEmbeddingDropout

> 
>      FullEmbeddingDropout (dropout:float)

_From https://github.com/jrzaurin/pytorch-widedeep/blob/be96b57f115e4a10fde9bb82c35380a3ac523f52/pytorch_widedeep/models/tab_transformer.py#L153_

* * *

source

### SharedEmbedding

> 
>      SharedEmbedding (num_embeddings, embedding_dim, shared_embed=True,
>                       add_shared_embed=False, shared_embed_div=8)

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

### ifnone

> 
>      ifnone (a, b)

_`b` if `a` is None else `a`_
    
    
    from fastai.tabular.all import *__
    
    
    path = untar_data(URLs.ADULT_SAMPLE)
    df = pd.read_csv(path/'adult.csv')
    dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
        cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'],
        cont_names = ['age', 'fnlwgt', 'education-num'],
        procs = [Categorify, FillMissing, Normalize])
    x_cat, x_cont, yb = first(dls.train)
    model = TabTransformer(dls.classes, dls.cont_names, dls.c)
    test_eq(model(x_cat, x_cont).shape, (dls.train.bs, dls.c))__

  * __Report an issue


