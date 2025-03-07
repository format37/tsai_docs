## On this page

  * Learner.get_X_preds



  * __Report an issue



# Inference

> Code required for inference.

* * *

source

### Learner.get_X_preds

> 
>      Learner.get_X_preds (X, y=None, bs=64, with_input=False,
>                           with_decoded=True, with_loss=False, act=None)

| **Type** | **Default** | **Details**  
---|---|---|---  
X |  |  |   
y | NoneType | None |   
bs | int | 64 |   
with_input | bool | False | returns the input as well  
with_decoded | bool | True | returns decoded predictions as well  
with_loss | bool | False | returns the loss per item as well  
act | NoneType | None | Apply activation to predictions, defaults to `self.loss_func`’s activation  
  
Get the predictions and targets, optionally with_input and with_loss.

with_decoded will also return the decoded predictions (it reverses the transforms applied).

The order of the output is the following:

  * input (optional): if with_input is True
  * **probabiblities** (for classification) or **predictions** (for regression)
  * **target** : if y is provided. Otherwise None.
  * **predictions** : predicted labels. Predictions will be decoded if with_decoded=True.
  * loss (optional): if with_loss is set to True and y is not None.


    
    
    from tsai.data.external import get_UCR_data __
    
    
    dsid = 'OliveOil'
    X, y, splits = get_UCR_data(dsid, split_data=False)
    X_test = X[splits[1]]
    y_test = y[splits[1]]__
    
    
    learn = load_learner("./models/test.pth")__

⚠️ Warning: load_learner (from fastai) requires all your custom code be in the exact same place as when exporting your Learner (the main script, or the module you imported it from).
    
    
    test_probas, test_targets, test_preds = learn.get_X_preds(X_test, with_decoded=True)
    test_probas, test_targets, test_preds __
    
    
    (tensor([[0.2574, 0.2421, 0.2364, 0.2641],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2421, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2421, 0.2364, 0.2641],
             [0.2574, 0.2422, 0.2364, 0.2639],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2641],
             [0.2574, 0.2421, 0.2364, 0.2641],
             [0.2574, 0.2421, 0.2364, 0.2640],
             [0.2574, 0.2421, 0.2364, 0.2641],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2421, 0.2364, 0.2641],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2421, 0.2364, 0.2641],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640]]),
     None,
     array(['4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4',
            '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4',
            '4', '4', '4', '4'], dtype='<U1'))
    
    
    import torch
    from fastcore.test import test_close __
    
    
    torch_test_probas, torch_test_targets, torch_test_preds = learn.get_X_preds(torch.from_numpy(X_test), with_decoded=True)
    torch_test_probas, torch_test_targets, torch_test_preds
    test_close(test_probas, torch_test_probas)__
    
    
    test_probas2, test_targets2, test_preds2 = learn.get_X_preds(X_test, y_test, with_decoded=True)
    test_probas2, test_targets2, test_preds2 __
    
    
    (tensor([[0.2574, 0.2421, 0.2364, 0.2641],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2421, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2421, 0.2364, 0.2641],
             [0.2574, 0.2422, 0.2364, 0.2639],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2641],
             [0.2574, 0.2421, 0.2364, 0.2641],
             [0.2574, 0.2421, 0.2364, 0.2640],
             [0.2574, 0.2421, 0.2364, 0.2641],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2421, 0.2364, 0.2641],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2421, 0.2364, 0.2641],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640]]),
     tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3]),
     array(['4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4',
            '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4',
            '4', '4', '4', '4'], dtype='<U1'))
    
    
    test_probas3, test_targets3, test_preds3, test_losses3 = learn.get_X_preds(X_test, y_test, with_loss=True, with_decoded=True)
    test_probas3, test_targets3, test_preds3, test_losses3 __
    
    
    (tensor([[0.2574, 0.2421, 0.2364, 0.2641],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2421, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2421, 0.2364, 0.2641],
             [0.2574, 0.2422, 0.2364, 0.2639],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2641],
             [0.2574, 0.2421, 0.2364, 0.2641],
             [0.2574, 0.2421, 0.2364, 0.2640],
             [0.2574, 0.2421, 0.2364, 0.2641],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2421, 0.2364, 0.2641],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2421, 0.2364, 0.2641],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640],
             [0.2574, 0.2422, 0.2364, 0.2640]]),
     tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3]),
     array(['4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4',
            '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4',
            '4', '4', '4', '4'], dtype='<U1'),
     TensorBase([1.3572, 1.3572, 1.3572, 1.3571, 1.3572, 1.4181, 1.4181, 1.4181,
                 1.4181, 1.4181, 1.4181, 1.4181, 1.4181, 1.4181, 1.4423, 1.4422,
                 1.4422, 1.4422, 1.3316, 1.3316, 1.3316, 1.3316, 1.3316, 1.3316,
                 1.3316, 1.3316, 1.3316, 1.3316, 1.3317, 1.3317]))
    
    
    from fastcore.test import test_eq __
    
    
    test_eq(test_probas, test_probas2)
    test_eq(test_preds, test_preds2)
    test_eq(test_probas, test_probas3)
    test_eq(test_preds, test_preds3)__

  * __Report an issue


