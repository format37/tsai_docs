## On this page

  * MatthewsCorrCoefBinary
  * get_task_metrics
  * F1_multi
  * Fbeta_multi
  * balanced_accuracy_multi
  * specificity_multi
  * recall_multi
  * precision_multi
  * metrics_multi_common
  * accuracy_multi
  * mae
  * mape



  * __Report an issue



  1. Training
  2. Metrics



# Metrics

> Metrics not included in fastai.

* * *

source

### MatthewsCorrCoefBinary

> 
>      MatthewsCorrCoefBinary (sample_weight=None)

_Matthews correlation coefficient for single-label classification problems_

* * *

source

### get_task_metrics

> 
>      get_task_metrics (dls, binary_metrics=None, multi_class_metrics=None,
>                        regression_metrics=None, verbose=True)

All metrics applicable to multi classification have been created by Doug Williams (https://github.com/williamsdoug). Thanks a lot Doug!!

* * *

source

### F1_multi

> 
>      F1_multi (*args, **kwargs)

* * *

source

### Fbeta_multi

> 
>      Fbeta_multi (inp, targ, beta=1.0, thresh=0.5, sigmoid=True)

_Computes Fbeta when`inp` and `targ` are the same size._

* * *

source

### balanced_accuracy_multi

> 
>      balanced_accuracy_multi (inp, targ, thresh=0.5, sigmoid=True)

_Computes balanced accuracy when`inp` and `targ` are the same size._

* * *

source

### specificity_multi

> 
>      specificity_multi (inp, targ, thresh=0.5, sigmoid=True)

_Computes specificity (true negative rate) when`inp` and `targ` are the same size._

* * *

source

### recall_multi

> 
>      recall_multi (inp, targ, thresh=0.5, sigmoid=True)

_Computes recall when`inp` and `targ` are the same size._

* * *

source

### precision_multi

> 
>      precision_multi (inp, targ, thresh=0.5, sigmoid=True)

_Computes precision when`inp` and `targ` are the same size._

* * *

source

### metrics_multi_common

> 
>      metrics_multi_common (inp, targ, thresh=0.5, sigmoid=True,
>                            by_sample=False)

_Computes TP, TN, FP, FN when`inp` and `targ` are the same size._

* * *

source

### accuracy_multi

> 
>      accuracy_multi (inp, targ, thresh=0.5, sigmoid=True, by_sample=False)

_Computes accuracy when`inp` and `targ` are the same size._

* * *

source

### mae

> 
>      mae (inp, targ)

_Mean absolute error between`inp` and `targ`._

* * *

source

### mape

> 
>      mape (inp, targ)

_Mean absolute percentage error between`inp` and `targ`._
    
    
    n_classes = 4
    inp = torch.normal(0, 1, (16, 20, n_classes))
    targ = torch.randint(0, n_classes, (16, 20)).to(torch.int8)
    _mAP(inp, targ)__
    
    
    0.27493315845795063

  * __Report an issue


