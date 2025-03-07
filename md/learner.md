## On this page

  * Learner.show_batch
  * Learner.remove_all_cbs
  * Learner.one_batch
  * Learner.inverse_transform
  * Learner.transform
  * load_all
  * Learner.save_all
  * Learner.plot_metrics
  * Recorder.plot_metrics
  * get_arch
  * ts_learner
  * tsimage_learner
  * Learner.decoder



  * __Report an issue



  1. Training
  2. Learner



# Learner

> fastai Learner extensions.

* * *

source

### Learner.show_batch

> 
>      Learner.show_batch (**kwargs)

* * *

source

### Learner.remove_all_cbs

> 
>      Learner.remove_all_cbs (max_iters=10)

* * *

source

### Learner.one_batch

> 
>      Learner.one_batch (i, b)

* * *

source

### Learner.inverse_transform

> 
>      Learner.inverse_transform (df:pandas.core.frame.DataFrame)

_Applies sklearn-type pipeline inverse transforms_

* * *

source

### Learner.transform

> 
>      Learner.transform (df:pandas.core.frame.DataFrame)

_Applies sklearn-type pipeline transforms_

⚠️ Important: save_all and load_all methods are designed for small datasets only. If you are using a larger dataset, you should use the standard save and load_learner methods.

* * *

source

### load_all

> 
>      load_all (path='export', dls_fname='dls', model_fname='model',
>                learner_fname='learner', device=None, pickle_module=<module
>                'pickle' from '/opt/hostedtoolcache/Python/3.10.16/x64/lib/pyth
>                on3.10/pickle.py'>, verbose=False)

* * *

source

### Learner.save_all

> 
>      Learner.save_all (path='export', dls_fname='dls', model_fname='model',
>                        learner_fname='learner', verbose=False)
    
    
    from tsai.data.core import get_ts_dls
    from tsai.utils import remove_dir __
    
    
    X = np.random.rand(100, 2, 10)
    dls = get_ts_dls(X)
    learn = Learner(dls, InceptionTimePlus(2, 1), loss_func=MSELossFlat())
    learn.save_all(Path.home()/'tmp', verbose=True)
    learn2 = load_all(Path.home()/'tmp', verbose=True)
    remove_dir(Path.home()/'tmp')__
    
    
    Learner saved:
    path          = '/Users/nacho/tmp'
    dls_fname     = '['dls_0.pth', 'dls_1.pth']'
    model_fname   = 'model.pth'
    learner_fname = 'learner.pkl'
    Learner loaded:
    path          = '/Users/nacho/tmp'
    dls_fname     = '['dls_0.pth', 'dls_1.pth']'
    model_fname   = 'model.pth'
    learner_fname = 'learner.pkl'
    /Users/nacho/tmp directory removed.

* * *

source

### Learner.plot_metrics

> 
>      Learner.plot_metrics (nrows:int=1, ncols:int=1, figsize:tuple=None,
>                            imsize:int=3, suptitle:str=None, sharex:"bool|Liter
>                            al['none','all','row','col']"=False, sharey:"bool|L
>                            iteral['none','all','row','col']"=False,
>                            squeeze:bool=True,
>                            width_ratios:Sequence[float]|None=None,
>                            height_ratios:Sequence[float]|None=None,
>                            subplot_kw:dict[str,Any]|None=None,
>                            gridspec_kw:dict[str,Any]|None=None)

| **Type** | **Default** | **Details**  
---|---|---|---  
nrows | int | 1 | Number of rows in returned axes grid  
ncols | int | 1 | Number of columns in returned axes grid  
figsize | tuple | None | Width, height in inches of the returned figure  
imsize | int | 3 | Size (in inches) of images that will be displayed in the returned figure  
suptitle | str | None | Title to be set to returned figure  
sharex | bool | Literal[‘none’, ‘all’, ‘row’, ‘col’] | False |   
sharey | bool | Literal[‘none’, ‘all’, ‘row’, ‘col’] | False |   
squeeze | bool | True |   
width_ratios | Sequence[float] | None | None |   
height_ratios | Sequence[float] | None | None |   
subplot_kw | dict[str, Any] | None | None |   
gridspec_kw | dict[str, Any] | None | None |   
**Returns** | **(plt.Figure, plt.Axes)** |  | **Returns both fig and ax as a tuple**  
  
* * *

source

### Recorder.plot_metrics

> 
>      Recorder.plot_metrics (nrows=None, ncols=None, figsize=None,
>                             final_losses=True, perc=0.5, imsize:int=3,
>                             suptitle:str=None, sharex:"bool|Literal['none','al
>                             l','row','col']"=False, sharey:"bool|Literal['none
>                             ','all','row','col']"=False, squeeze:bool=True,
>                             width_ratios:Sequence[float]|None=None,
>                             height_ratios:Sequence[float]|None=None,
>                             subplot_kw:dict[str,Any]|None=None,
>                             gridspec_kw:dict[str,Any]|None=None)

| **Type** | **Default** | **Details**  
---|---|---|---  
nrows | int | 1 | Number of rows in returned axes grid  
ncols | int | 1 | Number of columns in returned axes grid  
figsize | tuple | None | Width, height in inches of the returned figure  
final_losses | bool | True |   
perc | float | 0.5 |   
imsize | int | 3 | Size (in inches) of images that will be displayed in the returned figure  
suptitle | str | None | Title to be set to returned figure  
sharex | bool | Literal[‘none’, ‘all’, ‘row’, ‘col’] | False |   
sharey | bool | Literal[‘none’, ‘all’, ‘row’, ‘col’] | False |   
squeeze | bool | True |   
width_ratios | Sequence[float] | None | None |   
height_ratios | Sequence[float] | None | None |   
subplot_kw | dict[str, Any] | None | None |   
gridspec_kw | dict[str, Any] | None | None |   
**Returns** | **(plt.Figure, plt.Axes)** |  | **Returns both fig and ax as a tuple**  
  
* * *

source

### get_arch

> 
>      get_arch (arch_name)
    
    
    for arch_name in all_arch_names:
        get_arch(arch_name)__

* * *

source

### ts_learner

> 
>      ts_learner (dls, arch=None, c_in=None, c_out=None, seq_len=None, d=None,
>                  s_cat_idxs=None, s_cat_embeddings=None,
>                  s_cat_embedding_dims=None, s_cont_idxs=None, o_cat_idxs=None,
>                  o_cat_embeddings=None, o_cat_embedding_dims=None,
>                  o_cont_idxs=None, splitter=<function trainable_params>,
>                  loss_func=None, opt_func=<function Adam>, lr=0.001, cbs=None,
>                  metrics=None, path=None, model_dir='models', wd=None,
>                  wd_bn_bias=False, train_bn=True, moms=(0.95, 0.85, 0.95),
>                  train_metrics=False, valid_metrics=True, seed=None,
>                  device=None, verbose=False, patch_len=None,
>                  patch_stride=None, fusion_layers=128, fusion_act='relu',
>                  fusion_dropout=0.0, fusion_use_bn=True, pretrained=False,
>                  weights_path=None, exclude_head=True, cut=-1, init=None,
>                  arch_config={})

* * *

source

### tsimage_learner

> 
>      tsimage_learner (dls, arch=None, pretrained=False, loss_func=None,
>                       opt_func=<function Adam>, lr=0.001, cbs=None,
>                       metrics=None, path=None, model_dir='models', wd=None,
>                       wd_bn_bias=False, train_bn=True, moms=(0.95, 0.85,
>                       0.95), c_in=None, c_out=None, device=None,
>                       verbose=False, init=None, arch_config={})

* * *

source

### Learner.decoder

> 
>      Learner.decoder (o)
    
    
    from tsai.data.core import *
    from tsai.data.external import get_UCR_data
    from tsai.models.FCNPlus import FCNPlus __
    
    
    X, y, splits = get_UCR_data('OliveOil', verbose=True, split_data=False)
    tfms  = [None, [TSCategorize()]]
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms)
    learn = ts_learner(dls, FCNPlus)
    for p in learn.model.parameters():
        p.requires_grad=False
    test_eq(count_parameters(learn.model), 0)
    learn.freeze()
    test_eq(count_parameters(learn.model), 1540)
    learn.unfreeze()
    test_eq(count_parameters(learn.model), 264580)
    
    learn = ts_learner(dls, 'FCNPlus')
    for p in learn.model.parameters():
        p.requires_grad=False
    test_eq(count_parameters(learn.model), 0)
    learn.freeze()
    test_eq(count_parameters(learn.model), 1540)
    learn.unfreeze()
    test_eq(count_parameters(learn.model), 264580)__
    
    
    Dataset: OliveOil
    X      : (60, 1, 570)
    y      : (60,)
    splits : (#30) [0,1,2,3,4,5,6,7,8,9...] (#30) [30,31,32,33,34,35,36,37,38,39...] 
    
    
    
    learn.show_batch();__
    
    
    from fastai.metrics import accuracy
    from tsai.data.preprocessing import TSRobustScale __
    
    
    X, y, splits = get_UCR_data('OliveOil', split_data=False)
    tfms  = [None, TSClassification()]
    batch_tfms = TSRobustScale()
    dls = get_ts_dls(X, y, tfms=tfms, splits=splits, batch_tfms=batch_tfms)
    learn = ts_learner(dls, FCNPlus, metrics=accuracy, train_metrics=True)
    learn.fit_one_cycle(2)
    learn.plot_metrics()__

epoch | train_loss | train_accuracy | valid_loss | valid_accuracy | time  
---|---|---|---|---|---  
0 | 1.480875 | 0.266667 | 1.390461 | 0.300000 | 00:02  
1 | 1.476655 | 0.266667 | 1.387370 | 0.300000 | 00:01  
      
    
    if not os.path.exists("./models"): os.mkdir("./models")
    if not os.path.exists("./data"): os.mkdir("./data")
    np.save("data/X_test.npy", X[splits[1]])
    np.save("data/y_test.npy", y[splits[1]])
    learn.export("./models/test.pth")__

  * __Report an issue


