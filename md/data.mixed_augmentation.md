## On this page

  * MixHandler1d
  * MixUp1d
  * CutMix1d
  * IntraClassCutMix1d



  * __Report an issue



  1. Data
  2. Label-mixing transforms



# Label-mixing transforms

> Callbacks that perform data augmentation by mixing samples in different ways.

* * *

source

### MixHandler1d

> 
>      MixHandler1d (alpha=0.5)

_A handler class for implementing mixed sample data augmentation_

* * *

source

### MixUp1d

> 
>      MixUp1d (alpha=0.4)

_Implementation of https://arxiv.org/abs/1710.09412_
    
    
    from fastai.learner import *
    from tsai.models.InceptionTime import *
    from tsai.data.external import get_UCR_data
    from tsai.data.core import get_ts_dls, TSCategorize
    from tsai.data.preprocessing import TSStandardize
    from tsai.learner import ts_learner __
    
    
    X, y, splits = get_UCR_data('NATOPS', return_split=False)
    tfms = [None, TSCategorize()]
    batch_tfms = TSStandardize()
    dls = get_ts_dls(X, y, tfms=tfms, splits=splits, batch_tfms=batch_tfms)
    learn = ts_learner(dls, InceptionTime, cbs=MixUp1d(0.4))
    learn.fit_one_cycle(1)__

epoch | train_loss | valid_loss | time  
---|---|---|---  
0 | 1.908455 | 1.811908 | 00:03  
  
* * *

source

### CutMix1d

> 
>      CutMix1d (alpha=1.0)

_Implementation of`https://arxiv.org/abs/1905.04899`_

* * *

source

### IntraClassCutMix1d

> 
>      IntraClassCutMix1d (alpha=1.0)

_Implementation of CutMix applied to examples of the same class_
    
    
    X, y, splits = get_UCR_data('NATOPS', split_data=False)
    tfms = [None, TSCategorize()]
    batch_tfms = TSStandardize()
    dls = get_ts_dls(X, y, tfms=tfms, splits=splits, batch_tfms=batch_tfms)
    learn = ts_learner(dls, InceptionTime, cbs=IntraClassCutMix1d())
    learn.fit_one_cycle(1)__

epoch | train_loss | valid_loss | time  
---|---|---|---  
0 | 1.813483 | 1.792010 | 00:03  
      
    
    X, y, splits = get_UCR_data('NATOPS', split_data=False)
    tfms = [None, TSCategorize()]
    batch_tfms = TSStandardize()
    dls = get_ts_dls(X, y, tfms=tfms, splits=splits, batch_tfms=batch_tfms)
    learn = ts_learner(dls, cbs=CutMix1d(1.))
    learn.fit_one_cycle(1)__

epoch | train_loss | valid_loss | time  
---|---|---|---  
0 | 1.824509 | 1.774964 | 00:04  
  
  * __Report an issue


