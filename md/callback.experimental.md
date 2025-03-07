## On this page

  * Gambler’s loss: noisy labels
    * gambler_loss
    * GamblersCallback
  * Uncertainty-based data augmentation
    * UBDAug
  * BatchLossFilter
    * BatchLossFilter
  * RandomWeightLossWrapper
    * RandomWeightLossWrapper
  * BatchMasker
    * BatchMasker
  * Args:
  * SamplerWithReplacement
    * SamplerWithReplacement



  * __Report an issue



  1. Training
  2. Callbacks
  3. Experimental Callbacks



# Experimental Callbacks

> Miscellaneous experimental callbacks for timeseriesAI.

## Gambler’s loss: noisy labels

* * *

source

### gambler_loss

> 
>      gambler_loss (reward=2)

* * *

source

### GamblersCallback

> 
>      GamblersCallback (after_create=None, before_fit=None, before_epoch=None,
>                        before_train=None, before_batch=None, after_pred=None,
>                        after_loss=None, before_backward=None,
>                        after_cancel_backward=None, after_backward=None,
>                        before_step=None, after_cancel_step=None,
>                        after_step=None, after_cancel_batch=None,
>                        after_batch=None, after_cancel_train=None,
>                        after_train=None, before_validate=None,
>                        after_cancel_validate=None, after_validate=None,
>                        after_cancel_epoch=None, after_epoch=None,
>                        after_cancel_fit=None, after_fit=None)

_A callback to use metrics with gambler’s loss_
    
    
    from tsai.data.external import *
    from tsai.data.core import *
    from tsai.models.InceptionTime import *
    from tsai.models.layers import *
    from tsai.learner import *
    from fastai.metrics import *
    from tsai.metrics import *__
    
    
    X, y, splits = get_UCR_data('NATOPS', return_split=False)
    tfms = [None, TSCategorize()]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128])
    loss_func = gambler_loss()
    learn = ts_learner(dls, InceptionTime(dls.vars, dls.c + 1), loss_func=loss_func, cbs=GamblersCallback, metrics=[accuracy])
    learn.fit_one_cycle(1)__

epoch | train_loss | valid_loss | accuracy | time  
---|---|---|---|---  
0 | 1.840055 | 1.945397 | 0.166667 | 00:05  
  
## Uncertainty-based data augmentation

* * *

source

### UBDAug

> 
>      UBDAug (batch_tfms:list, N:int=2, C:int=4, S:int=1)

_A callback to implement the uncertainty-based data augmentation._
    
    
    from tsai.models.utils import *__
    
    
    X, y, splits = get_UCR_data('NATOPS', return_split=False)
    tfms = [None, TSCategorize()]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, batch_tfms=[TSStandardize()])
    model = build_ts_model(InceptionTime, dls=dls)
    TS_tfms = [TSMagScale(.75, p=.5), TSMagWarp(.1, p=0.5),  TSWindowWarp(.25, p=.5), 
               TSSmooth(p=0.5), TSRandomResizedCrop(.1, p=.5), 
               TSRandomCropPad(.3, p=0.5), 
               TSMagAddNoise(.5, p=.5)]
    
    ubda_cb = UBDAug(TS_tfms, N=2, C=4, S=2)
    learn = ts_learner(dls, model, cbs=ubda_cb, metrics=accuracy)
    learn.fit_one_cycle(1)__

epoch | train_loss | valid_loss | accuracy | time  
---|---|---|---|---  
0 | 1.817080 | 1.791119 | 0.077778 | 00:14  
  
# BatchLossFilter

* * *

source

### BatchLossFilter

> 
>      BatchLossFilter (loss_perc=1.0, schedule_func:Optional[<built-
>                       infunctioncallable>]=None)

_Callback that selects the hardest samples in every batch representing a percentage of the total loss_

# RandomWeightLossWrapper

* * *

source

### RandomWeightLossWrapper

> 
>      RandomWeightLossWrapper (after_create=None, before_fit=None,
>                               before_epoch=None, before_train=None,
>                               before_batch=None, after_pred=None,
>                               after_loss=None, before_backward=None,
>                               after_cancel_backward=None, after_backward=None,
>                               before_step=None, after_cancel_step=None,
>                               after_step=None, after_cancel_batch=None,
>                               after_batch=None, after_cancel_train=None,
>                               after_train=None, before_validate=None,
>                               after_cancel_validate=None, after_validate=None,
>                               after_cancel_epoch=None, after_epoch=None,
>                               after_cancel_fit=None, after_fit=None)

_Basic class handling tweaks of the training loop by changing a`Learner` in various events_

# BatchMasker

* * *

source

### BatchMasker

> 
>      BatchMasker (r:float=0.15, lm:int=3, stateful:bool=True, sync:bool=False,
>                   subsequence_mask:bool=True, variable_mask:bool=False,
>                   future_mask:bool=False, schedule_func:Optional[<built-
>                   infunctioncallable>]=None)

*Callback that applies a random mask to each sample in a training batch

# Args:

r: probability of masking. subsequence_mask: apply a mask to random subsequences. lm: average mask len when using stateful (geometric) masking. stateful: geometric distribution is applied so that average mask length is lm. sync: all variables have the same masking. variable_mask: apply a mask to random variables. Only applicable to multivariate time series. future_mask: used to train a forecasting model. schedule_func: if a scheduler is passed, it will modify the probability of masking during training.*

# SamplerWithReplacement

* * *

source

### SamplerWithReplacement

> 
>      SamplerWithReplacement (after_create=None, before_fit=None,
>                              before_epoch=None, before_train=None,
>                              before_batch=None, after_pred=None,
>                              after_loss=None, before_backward=None,
>                              after_cancel_backward=None, after_backward=None,
>                              before_step=None, after_cancel_step=None,
>                              after_step=None, after_cancel_batch=None,
>                              after_batch=None, after_cancel_train=None,
>                              after_train=None, before_validate=None,
>                              after_cancel_validate=None, after_validate=None,
>                              after_cancel_epoch=None, after_epoch=None,
>                              after_cancel_fit=None, after_fit=None)

_Callback that modify the sampler to select a percentage of samples and/ or sequence steps with replacement from each training batch_

  * __Report an issue


