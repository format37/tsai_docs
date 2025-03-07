## On this page

  * GatedTabTransformer



  * __Report an issue



  1. Models
  2. Tabular models
  3. GatedTabTransformer



# GatedTabTransformer

This implementation is based on:

  * Cholakov, R., & Kolev, T. (2022). **The GatedTabTransformer. An enhanced deep learning architecture for tabular modeling**. arXiv preprint arXiv:2201.00199. arXiv preprint https://arxiv.org/abs/2201.00199

  * Huang, X., Khetan, A., Cvitkovic, M., & Karnin, Z. (2020). **TabTransformer: Tabular Data Modeling Using Contextual Embeddings**. arXiv preprint https://arxiv.org/pdf/2012.06678




Official repo: https://github.com/radi-cho/GatedTabTransformer

* * *

source

### GatedTabTransformer

> 
>      GatedTabTransformer (classes, cont_names, c_out, column_embed=True,
>                           add_shared_embed=False, shared_embed_div=8,
>                           embed_dropout=0.1, drop_whole_embed=False,
>                           d_model=32, n_layers=6, n_heads=8, d_k=None,
>                           d_v=None, d_ff=None, res_attention=True,
>                           attention_act='gelu', res_dropout=0.1,
>                           norm_cont=True, mlp_d_model=32, mlp_d_ffn=64,
>                           mlp_layers=4)

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
    
    
    from fastcore.test import test_eq
    from fastcore.basics import first
    from fastai.data.external import untar_data, URLs
    from fastai.tabular.data import TabularDataLoaders
    from fastai.tabular.core import Categorify, FillMissing
    from fastai.data.transforms import Normalize
    import pandas as pd __
    
    
    path = untar_data(URLs.ADULT_SAMPLE)
    df = pd.read_csv(path/'adult.csv')
    dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
        cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'],
        cont_names = ['age', 'fnlwgt', 'education-num'],
        procs = [Categorify, FillMissing, Normalize])
    x_cat, x_cont, yb = first(dls.train)
    model = GatedTabTransformer(dls.classes, dls.cont_names, dls.c)
    test_eq(model(x_cat, x_cont).shape, (dls.train.bs, dls.c))__

  * __Report an issue


