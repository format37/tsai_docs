## On this page

  * MultiInputNet



  * __Report an issue



  1. Models
  2. Miscellaneous
  3. MultiInputNet



# MultiInputNet

This is an implementation created by Ignacio Oguiza (oguiza@timeseriesAI.co).

It can be used to combine different types of deep learning models into a single one that will accept multiple inputs from a MixedDataLoaders.

* * *

source

### MultiInputNet

> 
>      MultiInputNet (*models, c_out=None, reshape_fn=None, multi_output=False,
>                     custom_head=None, device=None, **kwargs)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    from tsai.basics import *
    from tsai.data.all import *
    from tsai.models.utils import *
    from tsai.models.InceptionTimePlus import *
    from tsai.models.TabModel import *__
    
    
    dsid = 'NATOPS'
    X, y, splits = get_UCR_data(dsid, split_data=False)
    ts_features_df = get_ts_features(X, y)__
    
    
    Feature Extraction: 100%|███████████████████████████████████████████| 40/40 [00:07<00:00,  5.23it/s]
    
    
    # raw ts
    tfms  = [None, [TSCategorize()]]
    batch_tfms = TSStandardize()
    ts_dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
    ts_model = build_ts_model(InceptionTimePlus, dls=ts_dls)
    
    # ts features
    cat_names = None
    cont_names = ts_features_df.columns[:-2]
    y_names = 'target'
    tab_dls = get_tabular_dls(ts_features_df, cat_names=cat_names, cont_names=cont_names, y_names=y_names, splits=splits)
    tab_model = build_tabular_model(TabModel, dls=tab_dls)
    
    # mixed
    mixed_dls = get_mixed_dls(ts_dls, tab_dls)
    MultiModalNet = MultiInputNet(ts_model, tab_model)
    learn = Learner(mixed_dls, MultiModalNet, metrics=[accuracy, RocAuc()])
    learn.fit_one_cycle(1, 1e-3)__

epoch | train_loss | valid_loss | accuracy | roc_auc_score | time  
---|---|---|---|---|---  
0 | 1.780674 | 1.571718 | 0.477778 | 0.857444 | 00:05  
      
    
    (ts, (cat, cont)),yb = mixed_dls.one_batch()
    learn.model((ts, (cat, cont))).shape __
    
    
    torch.Size([64, 6])
    
    
    tab_dls.c, ts_dls.c, ts_dls.cat __
    
    
    (6, 6, True)
    
    
    learn.loss_func __
    
    
    FlattenedLoss of CrossEntropyLoss()

  * __Report an issue


