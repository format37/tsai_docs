#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/07_Time_Series_Classification_with_Transformers.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# created by Ignacio Oguiza - email: oguiza@timeseriesAI.co

# # TST (Time Series Transformer) 🤗

# 
# > This is an unofficial PyTorch implementation by Ignacio Oguiza of  - oguiza@timeseriesAI.co based on:
# 
# * Zerveas, G., Jayaraman, S., Patel, D., Bhamidipaty, A., & Eickhoff, C. (2020). **A Transformer-based Framework for Multivariate Time Series Representation Learning**. arXiv preprint arXiv:2010.02803v2.
# * No official implementation available as far as I know (Oct 10th, 2020)
# 
# * This paper uses 'Attention is all you need' as a major reference:
#     * Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). **Attention is all you need**. In Advances in neural information processing systems (pp. 5998-6008).
# 
# This implementation is adapted to work with the rest of the `tsai` library, and contain some hyperparameters that are not available in the original implementation. I included them to experiment with them. 

# ## TST *args and **kwargs

# Usual values are the ones that appear in the "Attention is all you need" and "A Transformer-based Framework for Multivariate Time Series Representation Learning" papers. 
# 
# The default values are the ones selected as a default configuration in the latter.
# 
# * c_in: the number of features (aka variables, dimensions, channels) in the time series dataset. dls.var
# * c_out: the number of target classes. dls.c
# * seq_len: number of time steps in the time series. dls.len
# * max_seq_len: useful to control the temporal resolution in long time series to avoid memory issues. Default. None.
# * d_model: total dimension of the model (number of features created by the model). Usual values: 128-1024. Default: 128.
# * n_heads:  parallel attention heads. Usual values: 8-16. Default: 16.
# * d_k: size of the learned linear projection of queries and keys in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
# * d_v: size of the learned linear projection of values in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
# * d_ff: the dimension of the feedforward network model. Usual values: 256-4096. Default: 256.
# * dropout: amount of residual dropout applied in the encoder. Usual values: 0.-0.3. Default: 0.1.
# * activation: the activation function of intermediate layer, relu or gelu. Default: 'gelu'.
# * num_layers: the number of sub-encoder-layers in the encoder. Usual values: 2-8. Default: 3.
# * fc_dropout: dropout applied to the final fully connected layer. Usual values: 0-0.8. Default: 0.
# * pe: type of positional encoder. Available types: None, 'gauss' (default), 'lin1d', 'exp1d', '2d', 'sincos', 'zeros'. Default: 'gauss'.
# * learn_pe: learned positional encoder (True, default) or fixed positional encoder. Default: True.
# * flatten: this will flattent the encoder output to be able to apply an mlp type of head. Default=True.
# * custom_head: custom head that will be applied to the network. It must contain all kwargs (pass a partial function). Default: None.
# * y_range: range of possible y values (used in regression tasks). Default: None
# * kwargs: nn.Conv1d kwargs. If not {}, a nn.Conv1d with those kwargs will be applied to original time series.

# ## Tips on how to use transformers:

# * In general, transformers require a lower lr compared to other time series models when used with the same datasets. It's important to use `learn.lr_find()` to learn what a good lr may be. 
# 
# * The paper authors recommend to standardize data by feature. This can be done by adding `TSStandardize(by_var=True` as a batch_tfm when creating the `TSDataLoaders`.
# 
# * When using TST with a long time series, you may use `max_w_len` to reduce the memory size and thus avoid gpu issues.`
# 
# * I've tried different types of positional encoders. In my experience, the default one works just fine.
# 
# * In some of the cases I've used it, you may need to increase the dropout > .1 and/ or fc_dropout > 0 in order to achieve a good performance. 
# 
# * You may also experiment with other key hyperparameters like d_model, n_layers, n_heads, etc, but I have not seen major difference in my experience. 

# # Import libraries 📚

# In[ ]:


# # **************** UNCOMMENT AND RUN THIS CELL IF YOU NEED TO INSTALL/ UPGRADE TSAI ****************
# stable = True # Set to True for latest pip version or False for main branch in GitHub
# !pip install {"tsai -U" if stable else "git+https://github.com/timeseriesAI/tsai.git"} >> /dev/null


# In[ ]:


from tsai.all import *
computer_setup()


# # Load data 🔢

# In[ ]:


dsid = 'FaceDetection' 
X, y, splits = get_UCR_data(dsid, return_split=False)
print(X.shape, y.shape)


# # InceptionTime ⎘

# For comparison I will include a state-of-the-art time series model as Inception Time. 

# In[ ]:


bs = 64
n_epochs = 100
tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=bs, batch_tfms=TSStandardize())
dls.show_batch()


# In[ ]:


model = InceptionTime(dls.vars, dls.c)
learn = Learner(dls, model, metrics=[RocAucBinary(), accuracy], cbs=ShowGraphCallback2())
learn.lr_find()


# In[ ]:


start = time.time()
learn.fit_one_cycle(n_epochs, lr_max=1e-3)
print('\nElapsed time:', time.time() - start)


# We can see that even if valid loss goes up, the model doesn't overfit as there's no drop in performance.

# # TST baseline 🧢

# In[ ]:


bs = 64
tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=bs, batch_tfms=TSStandardize(by_var=True))
dls.show_batch()


# In[ ]:


model = TST(dls.vars, dls.c, dls.len)
learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropyFlat(), metrics=[RocAucBinary(), accuracy],  cbs=ShowGraphCallback2())
learn.lr_find()


# In[ ]:


model = TST(dls.vars, dls.c, dls.len)
learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropyFlat(), metrics=[RocAucBinary(), accuracy],  cbs=ShowGraphCallback2())
start = time.time()
learn.fit_one_cycle(n_epochs, lr_max=1e-4)
print('\nElapsed time:', time.time() - start)
learn.plot_metrics()


# # How to improve performance with TST? ➕

# The model clearly overfits in this task. To try and improve performance I will increase dropout. There are 2 types of dropout in TST: 
# 
# * applied to the MHAttention and Feed-Forward layers. Usually 0-0.3. Default: 0.1.
# 
# * applied to the fully connected head. Usually 0-0.8. Default: 0.
# 
# Let's see what's the impact of these 2 hyperparameters, used independently and combined.

# In[ ]:


model = TST(dls.vars, dls.c, dls.len, dropout=.3)
learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropyFlat(), metrics=[RocAucBinary(), accuracy],  cbs=ShowGraphCallback2())
start = time.time()
learn.fit_one_cycle(n_epochs, lr_max=1e-4)
print('\nElapsed time:', time.time() - start)
learn.plot_metrics()
beep()


# dropout by itself reduces overfit, but it doesn't eliminate it.

# In[ ]:


model = TST(dls.vars, dls.c, dls.len, dropout=.1, fc_dropout=.8)
learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropyFlat(), metrics=[RocAucBinary(), accuracy],  cbs=ShowGraphCallback2())
start = time.time()
learn.fit_one_cycle(n_epochs, lr_max=1e-4)
print('\nElapsed time:', time.time() - start)
learn.plot_metrics()
beep()


# It still slightly overfits, although it's much better than the original settings. 
# 
# Now let's try both together.

# In[ ]:


model = TST(dls.vars, dls.c, dls.len, dropout=.3, fc_dropout=.8)
learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropyFlat(), metrics=[RocAucBinary(), accuracy],  cbs=ShowGraphCallback2())
start = time.time()
learn.fit_one_cycle(n_epochs, lr_max=1e-4)
print('\nElapsed time:', time.time() - start)
learn.plot_metrics()
beep()


# Let's check what happens if we increase dropout a bit more...

# In[ ]:


model = TST(dls.vars, dls.c, dls.len, dropout=0.3, fc_dropout=0.9)
learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropyFlat(), metrics=[RocAucBinary(), accuracy],  cbs=ShowGraphCallback2())
learn.fit_one_cycle(n_epochs, 1e-4) 
learn.plot_metrics()
beep()


# This is a great result, beyond InceptionTime and any other the state-of-the-art papers I've seen.

# # Conclusion ✅

# TST (Time Series Transformer) seems like a great addition to the world of time series models.
# 
# The model trains very smoothly and overfitting can be reduced/ eliminated by using dropout.
# 
# Also, TST is about 10% faster to train that InceptionTime.
# Here's all the code you need to train a transformer model with `tsai`:
# 
# ```
# X, y, splits = get_UCR_data('FaceDetection', return_split=False)
# tfms  = [None, [Categorize()]]
# dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
# dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=64, batch_tfms=TSStandardize(by_var=True))
# model = TST(dls.vars, dls.c, dls.len, dropout=0.3, fc_dropout=0.9)
# learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropyFlat(), 
#                 metrics=[RocAucBinary(), accuracy],  cbs=ShowGraphCallback2())
# learn.fit_one_cycle(100, 1e-4) 
# ```
