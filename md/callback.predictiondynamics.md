## On this page

  * PredictionDynamics



  * __Report an issue



  1. Training
  2. Callbacks
  3. PredictionDynamics



# PredictionDynamics

> Callback used to visualize model predictions during training.

This is an implementation created by Ignacio Oguiza (oguiza@timeseriesAI.co) based on a blog post by Andrej Karpathy I read some time ago that I really liked. One of the things he mentioned was this:

> “**visualize prediction dynamics**. I like to visualize model predictions on a fixed test batch during the course of training. The “dynamics” of how these predictions move will give you incredibly good intuition for how the training progresses. Many times it is possible to feel the network “struggle” to fit your data if it wiggles too much in some way, revealing instabilities. Very low or very high learning rates are also easily noticeable in the amount of jitter.” A. Karpathy

* * *

source

### PredictionDynamics

> 
>      PredictionDynamics (show_perc=1.0, figsize=(10, 6), alpha=0.3, size=30,
>                          color='lime', cmap='gist_rainbow', normalize=False,
>                          sensitivity=None, specificity=None)

_Basic class handling tweaks of the training loop by changing a`Learner` in various events_
    
    
    from tsai.basics import *
    from tsai.models.InceptionTime import *__
    
    
    dsid = 'NATOPS'
    X, y, splits = get_UCR_data(dsid, split_data=False)
    check_data(X, y, splits, False)__
    
    
    X      - shape: [360 samples x 24 features x 51 timesteps]  type: memmap  dtype:float32  isnan: 0
    y      - shape: (360,)  type: memmap  dtype:<U3  n_classes: 6 (60 samples per class) ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0']  isnan: False
    splits - n_splits: 2 shape: [180, 180]  overlap: False
    
    
    tfms  = [None, [Categorize()]]
    batch_tfms = [TSStandardize(by_var=True)]
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
    learn = ts_learner(dls, InceptionTime, metrics=accuracy, cbs=PredictionDynamics()) 
    learn.fit_one_cycle(2, 3e-3)__

epoch | train_loss | valid_loss | accuracy | time  
---|---|---|---|---  
0 | 1.885462 | 1.773872 | 0.238889 | 00:05  
1 | 1.425667 | 1.640418 | 0.377778 | 00:05  
  
| train_loss | valid_loss | accuracy  
---|---|---|---  
1 | 1.425667 | 1.640418 | 0.377778  
  
  * __Report an issue


