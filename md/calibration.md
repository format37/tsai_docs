## On this page

  * ECELoss
  * TemperatureSetter
  * ModelWithTemperature
  * plot_calibration_curve
  * Learner.calibrate_model



  * __Report an issue



  1. Training
  2. Calibration



# Calibration

> Functionality to calibrate a trained, binary classification model using temperature scaling.

* * *

source

### ECELoss

> 
>      ECELoss (n_bins=10)

_Calculates the Expected Calibration Error of a model._

* * *

source

### TemperatureSetter

> 
>      TemperatureSetter (model, lr=0.01, max_iter=1000, line_search_fn=None,
>                         n_bins=10, verbose=True)

_Calibrates a binary classification model optimizing temperature_

* * *

source

### ModelWithTemperature

> 
>      ModelWithTemperature (model)

_A decorator which wraps a model with temperature scaling_

* * *

source

### plot_calibration_curve

> 
>      plot_calibration_curve (labels, logits, cal_logits=None, figsize=(6, 6),
>                              n_bins=10, strategy='uniform')

* * *

source

### Learner.calibrate_model

> 
>      Learner.calibrate_model (X=None, y=None, lr=0.01, max_iter=10000,
>                               line_search_fn=None, n_bins=10,
>                               strategy='uniform', show_plot=True, figsize=(6,
>                               6), verbose=True)
    
    
    from tsai.basics import *__
    
    
    X, y, splits = get_UCR_data('FingerMovements', split_data=False)
    tfms  = [None, TSClassification()]
    batch_tfms = TSRobustScale()
    # dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
    learn = TSClassifier(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms,
                         arch='FCNPlus', metrics=accuracy)
    learn.fit_one_cycle(2)__

epoch | train_loss | valid_loss | accuracy | time  
---|---|---|---|---  
0 | 0.794469 | 0.731429 | 0.500000 | 00:00  
1 | 0.758404 | 0.716087 | 0.490000 | 00:00  
      
    
    learn.calibrate_model()
    calibrated_model = learn.calibrated_model __
    
    
    Before temperature - NLL: 0.716, ECE: 0.093
    Calibrating the model...
    ...model calibrated
    Optimal temperature: 272.026
    After temperature  - NLL: 0.693, ECE: 0.010
    

  * __Report an issue


