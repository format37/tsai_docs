#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/08_Self_Supervised_MVP.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# created by Ignacio Oguiza - email: oguiza@timeseriesAI.co

# # MVP (previously TSBERT): Self-Supervised Pretraining of Time Series Models 🤗

# This is an unofficial PyTorch implementation created by Ignacio Oguiza (oguiza@timeseriesAI.co) based on:
# 
# * Zerveas, G., Jayaraman, S., Patel, D., Bhamidipaty, A., & Eickhoff, C. (2020). A Transformer-based Framework for Multivariate Time Series Representation Learning. arXiv preprint arXiv:2010.02803v2.. No official implementation available as far as I know (Oct 10th, 2020)

# `MVP` is a self-supervised training method that can be used to pretrain time series models without using any labels. The approach is very similar to BERT.
# 
# `MVP` is performed in 2 steps: 
# 
# 1. Pretrain the selected architecture without any labels. When training is finished, the pretrained model will be automatically saved to the given target_dir/fname.
# 2. Fine-tune or train the same architecture with pretrained=True indicating the weights_path (target_dir/fname).
# 
# 
# In this notebook we'll use a UCR dataset (LSST) that contains around 2500 training and 2500 validation samples. To analyze the impact of `MVP` we'll:
# 1. use supervised learning to set a baseline using 10% or 100% of the labels.
# 2. pretrain a model using 100% of the training dataset without labels.
# 3. fine tune or train using 10% or 100% of the training dataset (with labels). 
# 
# A key difference between `MVP` and the original paper is that you can use any architecture of your choice as long as it has a "head" attribute and can take a custom_head kwarg. Architectures finished in Plus in the `tsai` library meet this criteria. To demonstrate how this works, we'll use InceptionTimePlus throughout this notebook.

# ### Results

# <img src="https://github.com/timeseriesAI/tsai/blob/master/tutorial_nbs/images/TSBERT_data.jpg?raw=1">
# <img src="https://github.com/timeseriesAI/tsai/blob/master/tutorial_nbs/images/TSBERT_chart.jpg?raw=1">

# These results indicate the following: 
# 
# * Pretraining + fine-tuning/ training improves performance when compared to supervised learning (training from scratch).
# * In this case, there's not much difference between fine-tuning and training a pretrained model. This may be dataset dependent. It'd be good to try both approaches. 
# * The fewer labels available, the better pretraining seems to work. 

# # Import libraries 📚

# In[ ]:


# # **************** UNCOMMENT AND RUN THIS CELL IF YOU NEED TO INSTALL/ UPGRADE TSAI ****************
# stable = True # Set to True for latest pip version or False for main branch in GitHub
# !pip install {"tsai -U" if stable else "git+https://github.com/timeseriesAI/tsai.git"} >> /dev/null


# In[ ]:


from tsai.all import *
from IPython.display import clear_output
my_setup()


# # Prepare data 🏭

# We'll first import the data.

# In[ ]:


dsid = 'LSST'
X, y, splits = get_UCR_data(dsid, split_data=False)


# We'll now create 2 dataloaders with 100% of the training and 100% of validation samples.
# One of them doesn't contain the y (unlabeled). The other one contains the labels. 
# We'll use the unlabeled dataset (udls) to pretrain the model.

# In[ ]:


# 100% train data
tfms = [None, TSClassification()]
batch_tfms = [TSStandardize(by_sample=True)]
check_data(X, y, splits)
dls100 = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
udls100 = get_ts_dls(X, splits=splits, tfms=tfms, batch_tfms=batch_tfms) # used in pretraining


# We'll also need a labeled dataloaders with 10% of the training and 100% of validation data.

# In[ ]:


# 10% train data
train_split010 = get_splits(y[splits[0]], valid_size=.1, show_plot=False)[1]
splits010 = (train_split010, splits[1])
check_data(X, y, splits010)
dls010 = get_ts_dls(X, y, splits=splits010, tfms=tfms, batch_tfms=batch_tfms)


# # Supervised 👀

# First we'll train a model in a supervised way to set a baseline. We'll train using 10% and 100% of the training set. We'll run 10 tests.
# 
# We'll train all models with the same settings (50 epochs with 10% of labels, 20 and 50 epochs with 100% of labels and lr=1e-2) to see the impact of pretraining.

# In[ ]:


# supervised 10%
n_epochs = 50
n_tests = 10
_result = []
for i in range(n_tests):
    clear_output()
    if i > 0: print(f'{i}/{n_tests} accuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f}')
    else: print(f'{i}/{n_tests}')
    learn = ts_learner(dls010, InceptionTimePlus, metrics=accuracy)
    learn.fit_one_cycle(n_epochs, 1e-2)
    _result.append(learn.recorder.values[-1][-1])
learn.plot_metrics()
print(f'\naccuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f} in {n_tests} tests')


# In[ ]:


# supervised 100%
n_epochs = 50
n_tests = 10
_result = []
for i in range(n_tests):
    clear_output()
    if i > 0: print(f'{i}/{n_tests} accuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f}')
    else: print(f'{i}/{n_tests}')
    learn = ts_learner(dls100, InceptionTimePlus, metrics=accuracy)
    learn.fit_one_cycle(n_epochs, 1e-2)
    _result.append(learn.recorder.values[-1][-1])
learn.plot_metrics()
print(f'\naccuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f} in {n_tests} tests')


# I've also trained the model with all labels for 20 as it seems to be overfitting with 50 epochs.

# In[ ]:


# supervised 100%
n_epochs = 20
n_tests = 10
_result = []
for i in range(n_tests):
    clear_output()
    if i > 0: print(f'{i}/{n_tests} accuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f}')
    else: print(f'{i}/{n_tests}')
    learn = ts_learner(dls100, InceptionTimePlus, metrics=accuracy)
    learn.fit_one_cycle(n_epochs, 1e-2)
    _result.append(learn.recorder.values[-1][-1])
learn.plot_metrics()
print(f'\naccuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f} in {n_tests} tests')


# This is a great result. Just for reference, in a recent review of multivariate time series models (Ruiz, A. P., Flynn, M., Large, J., Middlehurst, M., & Bagnall, A. (2020). The great multivariate time series classification bake off: a review and experimental evaluation of recent algorithmic advances. Data Mining and Knowledge Discovery, 1-49.), the best performing classifier on this dataset is MUSE with an accuracy of **63.62**. 
# 
# Let's see if we can improve our baseline pretraining InceptionTime using `MVP`.

# # Pretrain model  🏋️‍♂️

# Now we'll train a model without any labels on the entire training set. To do that we need to use the `MVP` callback. You can get more details on this callback visiting [`tsai` documentation](https://timeseriesai.github.io/tsai/callback.MVP).

# In[ ]:


# Unlabeled 100%
learn = ts_learner(udls100, InceptionTimePlus, cbs=[ShowGraph(), MVP(target_dir='./data/MVP', fname=f'{dsid}_200')])
learn.fit_one_cycle(200, 1e-2)


# In[ ]:


learn.MVP.show_preds(sharey=True)


# # Fine-tune 🎻

# There are at least 2 options to use the pretrained model weights: 
# 
# 1.   Fine-tune
# 2.   Train
# 
# 
# We'll start by fine-tuning the pretrained model.
# 
# In this case, we double the base_lr as the training lr will be the base_lr / 2. The only net change of the fine tuning then is just a training of the new head for 10 epochs. The rest of the training will be the same.
# 
# Before training though, we'll check that when the model is frozen only the last layer is trained: 

# In[ ]:


learn = ts_learner(dls010, InceptionTimePlus, pretrained=True, weights_path=f'data/MVP/{dsid}_200.pth', metrics=accuracy)
for p in learn.model.parameters():
    p.requires_grad=False
print(f'{"trainable params once manually frozen":40}: {count_parameters(learn.model):8}')
learn.freeze()
print(f'{"trainable params after learn.freeze()":40}: {count_parameters(learn.model):8}')
learn.unfreeze()
print(f'{"trainable params learn.unfreeze()":40}: {count_parameters(learn.model):8}')


# It seems to be working well.

# In[ ]:


# self-supervised: fine-tuning with 10% labels
n_epochs = 50
freeze_epochs = 10
n_tests = 10
_result = []
for i in range(n_tests):
    clear_output()
    if i > 0: print(f'{i}/{n_tests} accuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f}')
    else: print(f'{i}/{n_tests}')
    learn = ts_learner(dls010, InceptionTimePlus, pretrained=True, weights_path=f'data/MVP/{dsid}_200.pth', metrics=accuracy)
    learn.fine_tune(n_epochs, base_lr=2e-2, freeze_epochs=freeze_epochs)
    _result.append(learn.recorder.values[-1][-1])
learn.plot_metrics()
print(f'\naccuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f} in {n_tests} tests')


# In[ ]:


# self-supervised: fine-tuning with 100% labels
n_epochs = 50
freeze_epochs = 10
n_tests = 10
_result = []
for i in range(n_tests):
    clear_output()
    if i > 0: print(f'{i}/{n_tests} accuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f}')
    else: print(f'{i}/{n_tests}')
    learn = ts_learner(dls100, InceptionTimePlus, pretrained=True, weights_path=f'data/MVP/{dsid}_200.pth', metrics=accuracy)
    learn.fine_tune(n_epochs, base_lr=2e-2, freeze_epochs=freeze_epochs)
    _result.append(learn.recorder.values[-1][-1])
learn.plot_metrics()
print(f'\naccuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f} in {n_tests} tests')


# In[ ]:


# self-supervised: fine-tuning with 100% labels
n_epochs = 20
freeze_epochs = 10
n_tests = 10
_result = []
for i in range(n_tests):
    clear_output()
    if i > 0: print(f'{i}/{n_tests} accuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f}')
    else: print(f'{i}/{n_tests}')
    learn = ts_learner(dls100, InceptionTimePlus, pretrained=True, weights_path=f'data/MVP/{dsid}_200.pth', metrics=accuracy)
    learn.fine_tune(n_epochs, base_lr=2e-2, freeze_epochs=freeze_epochs)
    _result.append(learn.recorder.values[-1][-1])
learn.plot_metrics()
print(f'\naccuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f} in {n_tests} tests')


# # Train 🏃🏽‍♀️🏃🏽‍♀

# In[ ]:


# self-supervised: train with 10% labels
n_epochs = 50
n_tests = 10
_result = []
for i in range(n_tests):
    clear_output()
    if i > 0: print(f'{i}/{n_tests} accuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f}')
    else: print(f'{i}/{n_tests}')
    learn = ts_learner(dls010, InceptionTimePlus, pretrained=True, weights_path=f'data/MVP/{dsid}_200.pth', metrics=accuracy)
    learn.fit_one_cycle(n_epochs, 1e-2)
    _result.append(learn.recorder.values[-1][-1])
learn.plot_metrics()
print(f'\naccuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f} in {n_tests} tests')


# In[ ]:


# self-supervised: train with 100% labels
n_epochs = 50
n_tests = 10
_result = []
for i in range(n_tests):
    clear_output()
    if i > 0: print(f'{i}/{n_tests} accuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f}')
    else: print(f'{i}/{n_tests}')
    learn = ts_learner(dls100, InceptionTimePlus, pretrained=True, weights_path=f'data/MVP/{dsid}_200.pth', metrics=accuracy)
    learn.fit_one_cycle(n_epochs, 1e-2)
    _result.append(learn.recorder.values[-1][-1])
learn.plot_metrics()
print(f'\naccuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f} in {n_tests} tests')


# In[ ]:


# self-supervised 100% + training
n_epochs = 20
n_tests = 10
_result = []
for i in range(n_tests):
    clear_output()
    if i > 0: print(f'{i}/{n_tests} accuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f}')
    else: print(f'{i}/{n_tests}')
    learn = ts_learner(dls100, InceptionTimePlus, pretrained=True, weights_path=f'data/MVP/{dsid}_200.pth', metrics=accuracy)
    learn.fit_one_cycle(n_epochs, 1e-2)
    _result.append(learn.recorder.values[-1][-1])
learn.plot_metrics()
print(f'\naccuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f} in {n_tests} tests')


# # Adding data augmentation 🔎🔎

# One last thing I'd like to test is the impact of data augmentation when using a pretrained model. 
# 
# I will compare the performance of a model trained from scratch and a pretrained model adding CutMix (CutMix1D in `tsai`). We'll see if the difference in performance still holds.

# In[ ]:


# self-supervised 100% + training
n_epochs = 20
n_tests = 10
_result = []
for i in range(n_tests):
    clear_output()
    if i > 0: print(f'{i}/{n_tests} accuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f}')
    else: print(f'{i}/{n_tests}')
    learn = ts_learner(dls100, InceptionTimePlus, metrics=accuracy, cbs=CutMix1d())
    learn.fit_one_cycle(n_epochs, 1e-2)
    _result.append(learn.recorder.values[-1][-1])
learn.plot_metrics()
print(f'\naccuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f} in {n_tests} tests')


# In[ ]:


# self-supervised 100% + training with cutmix
n_epochs = 20
n_tests = 10
_result = []
for i in range(n_tests):
    clear_output()
    if i > 0: print(f'{i}/{n_tests} accuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f}')
    else: print(f'{i}/{n_tests}')
    learn = ts_learner(dls100, InceptionTimePlus, pretrained=True, weights_path=f'data/MVP/{dsid}_200.pth', metrics=accuracy, cbs=CutMix1d())
    learn.fit_one_cycle(n_epochs, 1e-2)
    _result.append(learn.recorder.values[-1][-1])
learn.plot_metrics()
print(f'\naccuracy: {np.mean(_result):.3f} +/- {np.std(_result):.3f} in {n_tests} tests')


# As you can see CutMix improves performance in both cases, but the pretrained model still performs better.

# # Conclusions ✅

# `MVP` is the first self-supervised method added to the `tsai` library. And it seems to work pretty well. It shows something really interesting: self-supervised learning may improve performance, with or without additional unlabeled data.
# 
# In this notebook we've demonstrated how easy it is to use a self-supervised method in 2 steps: 
# 
# 1. pretrain an architecture using the `MVP` callback.
# 2. fine-tune or train the same architecture using the pretrained model weights.
# 
# In all cases, performance has been better when using the pretrained model weights as a starting point. 
# 
# In this case, training has proven to be superior to fine-tuning. However, this will not always be the case. It's difficult to know a priori whether fine/tuning or training will provide better results. I'd recommend trying both approaches. 
# 
# In the case of using data augmentation (data augmentation), we've also seen that the pretrained model performs better than the one trained from scratch.
# 
# `MVP` has shown it can improve performance with a low number of labels (10%) as well as with all labels (100%). 
# 
# I'd encourage you to use `MVP` with your own datasets and share your experience.
