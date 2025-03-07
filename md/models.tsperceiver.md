## On this page

  * TSPerceiver



  * __Report an issue



  1. Models
  2. Transformers
  3. TSPerceiver



# TSPerceiver

This implementation is inspired by:

Jaegle, A., Gimeno, F., Brock, A., Zisserman, A., Vinyals, O., & Carreira, J. (2021).

**Perceiver: General Perception with Iterative Attention**. arXiv preprint arXiv:2103.03206.

Paper: https://arxiv.org/pdf/2103.03206.pdf

Official repo: Not available as og April, 2021.

* * *

source

### TSPerceiver

> 
>      TSPerceiver (c_in, c_out, seq_len, cat_szs=0, n_cont=0, n_latents=512,
>                   d_latent=128, d_context=None, n_layers=6,
>                   self_per_cross_attn=1, share_weights=True, cross_n_heads=1,
>                   self_n_heads=8, d_head=None, attn_dropout=0.0,
>                   fc_dropout=0.0, concat_pool=False)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    from tsai.basics import *
    from tsai.data.all import *__
    
    
    dsid = 'OliveOil'
    X, y, splits = get_UCR_data(dsid, split_data=False)
    ts_features_df = get_ts_features(X, y)
    ts_features_df.shape __
    
    
    Feature Extraction: 100%|██████████████████████████████████████████| 30/30 [00:00<00:00, 189.16it/s]
    
    
    (60, 11)
    
    
    # raw ts
    tfms  = [None, [Categorize()]]
    batch_tfms = TSStandardize(by_sample=True)
    ts_dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
    
    # ts features
    cat_names = None
    cont_names = ts_features_df.columns[:-2]
    y_names = 'target'
    tab_dls = get_tabular_dls(ts_features_df, cat_names=cat_names, cont_names=cont_names, y_names=y_names, splits=splits)
    
    # mixed
    mixed_dls = get_mixed_dls(ts_dls, tab_dls)
    xb, yb = mixed_dls.one_batch()__
    
    
    model = TSPerceiver(ts_dls.vars, ts_dls.c, ts_dls.len, cat_szs=0, 
                        # n_cont=0, 
                        n_cont=xb[1][1].shape[1], 
                        n_latents=128, d_latent=128, n_layers=3, self_per_cross_attn=1, share_weights=True,
                        cross_n_heads=16, self_n_heads=16, d_head=None, attn_dropout=0., fc_dropout=0.).to(device)
    test_eq(model(xb).shape, (yb.shape[0], len(np.unique(y))))__

  * __Report an issue


