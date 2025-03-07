## On this page

  * TabFusionTransformer
  * TabFusionBackbone
  * Sequential
  * ifnone
  * TSTabFusionTransformer



  * __Report an issue



  1. Models
  2. Tabular models
  3. TabFusionTransformer



# TabFusionTransformer

This is a a Pytorch implementeation of TabTransformerTransformer created by Ignacio Oguiza (oguiza@timeseriesAI.co)

This implementation is inspired by:

Huang, X., Khetan, A., Cvitkovic, M., & Karnin, Z. (2020). **TabTransformer: Tabular Data Modeling Using Contextual Embeddings**. arXiv preprint https://arxiv.org/pdf/2012.06678

Official repo: https://github.com/awslabs/autogluon/tree/master/tabular/src/autogluon/tabular/models/tab_transformer

* * *

source

### TabFusionTransformer

> 
>      TabFusionTransformer (classes, cont_names, c_out, d_model=32, n_layers=6,
>                            n_heads=8, d_k=None, d_v=None, d_ff=None,
>                            res_attention=True, attention_act='gelu',
>                            res_dropout=0.0, fc_mults=(4, 2), fc_dropout=0.0,
>                            fc_act=None, fc_skip=False, fc_bn=False,
>                            bn_final=False, init=True)

_Class that allows you to pass one or multiple inputs_

* * *

source

### TabFusionBackbone

> 
>      TabFusionBackbone (classes, cont_names, d_model=32, n_layers=6,
>                         n_heads=8, d_k=None, d_v=None, d_ff=None, init=True,
>                         res_attention=True, attention_act='gelu',
>                         res_dropout=0.0)

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

### Sequential

> 
>      Sequential (*args)

_Class that allows you to pass one or multiple inputs_

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
    model = TabFusionTransformer(dls.classes, dls.cont_names, dls.c)
    test_eq(model(x_cat, x_cont).shape, (dls.train.bs, dls.c))__

* * *

source

### TSTabFusionTransformer

> 
>      TSTabFusionTransformer (c_in, c_out, seq_len, classes, cont_names,
>                              d_model=32, n_layers=6, n_heads=8, d_k=None,
>                              d_v=None, d_ff=None, res_attention=True,
>                              attention_act='gelu', res_dropout=0.0,
>                              fc_mults=(1, 0.5), fc_dropout=0.0, fc_act=None,
>                              fc_skip=False, fc_bn=False, bn_final=False,
>                              init=True)

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
    
    
    classes = {'education': ['#na#', '10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 
                             'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college'],
     'education-num_na': ['#na#', False, True],
     'marital-status': ['#na#', 'Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'],
     'occupation': ['#na#', '?', 'Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 
                    'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'],
     'race': ['#na#', 'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'],
     'relationship': ['#na#', 'Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife'],
     'workclass': ['#na#', '?', 'Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay']}
    
    cont_names = ['a', 'b', 'c']
    c_out = 6
    x_ts = torch.randn(64, 3, 10)
    x_cat = torch.randint(0,3,(64,7))
    x_cont = torch.randn(64,3)
    model = TSTabFusionTransformer(x_ts.shape[1], c_out, x_ts.shape[-1], classes, cont_names)
    x = (x_ts, (x_cat, x_cont))
    test_eq(model(x).shape, (x_ts.shape[0], c_out))__

  * __Report an issue


