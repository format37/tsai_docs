#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/01_Intro_to_Time_Series_Classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# created by Ignacio Oguiza - email: oguiza@timeseriesAI.co

# ## Purpose 😇

# The purpose of this notebook is to show you how you can create a simple, end-to-end, state-of-the-art time series classification model using the great **fastai-v2** library in 5 steps:
# 1. Import libraries
# 2. Prepare data
# 3. Build learner
# 4. Train model
# 5. Inference (predictions) on additional data
# 
# In general, there are 3 main ways to classify time series, based on the input to the neural network:
# 
# - raw data
# 
# - image data (encoded from raw data)
# 
# - feature data (extracted from raw data)
# 
# In this notebook, we will use the first approach.

# ## Import libraries 📚

# In[ ]:


# # **************** UNCOMMENT AND RUN THIS CELL IF YOU NEED TO INSTALL/ UPGRADE TSAI ****************
# stable = True # Set to True for latest pip version or False for main branch in GitHub
# !pip install {"tsai -U" if stable else "git+https://github.com/timeseriesAI/tsai.git"} >> /dev/null


# In[ ]:


from tsai.all import *
import sklearn.metrics as skm
my_setup()


# ## Prepare data 🔢

# ### Download data ⬇️

# In this notebook, we'll use one of the most widely used time series classification databases: UEA & UCR Time Series Classification Repository. As of Sep 2019 it contains 128 univariate datasets and 30 multivariate datasets.
# 

# In[ ]:


print(get_UCR_univariate_list())


# In[ ]:


print(get_UCR_multivariate_list())


# In the case of UCR data it's very easy to get data loaded. Let's select a dataset. You can modify this and select any one from the previous lists (univariate of multivariate).
# 
# `return_split` determines whether the UCR data will be returned already split between train and test or not.

# In[ ]:


# dataset id
dsid = 'NATOPS' 
X, y, splits = get_UCR_data(dsid, return_split=False)
X.shape, y.shape, splits


# ☣️ **Something very important when you prepare your own data is that data needs to be in a 3-d array with the following format:**
# 
# 1. Samples
# 2. Variables
# 3. Length (aka time or sequence steps)
# 
# Variables = 1 for univariate datasets and >1 for multivariate.
# 
# In the case your data is already separate between train and test like this:

# In[ ]:


X_train, y_train, X_test, y_test  = get_UCR_data(dsid, return_split=True)


# you can use this convenience function to get X, y and splits:

# In[ ]:


X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])


# All UEA & UCR Time Series Classification data have already been split between train and valid. When you use your own data, you'll have to split it yourself. We'll see examples of this in future notebooks.

# ### Prepare datasets 💿

# The first step is to create datasets. This is very easy to do in v2. 
# 
# In TS classification problems, you will usually want to use an item tfm to transform y into categories.
# 
# We'll use `inplace=True` to preprocess data at dataset initialization. This will significantly speed up training. 

# In[ ]:


tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)


# We'll now build the `DataLoader`s (dls) that will create batches of data.
# 
# You will need to pass:
# 
# * datasets: usually 2 - train and valid -  or 1 - test or unlabelled- depending on the problem
# * batch size(s): you may pass a single value (will be applied to all dls, or different values, one for each dl.
# * batch_tfms (same as after_batch): you may decide to pass some tfms at the batch level. In this case for example, we'll standardize the data (0 mean and 1 std). You may get more details on how these transforms work in the transforms nb.
# * num workers: num_workers > 0 is used to preprocess batches of data so that the next batch is ready for use when the current batch has been finished. More num_workers would consume more memory usage but is helpful to speed up the I/O process. This will depend on your machine, dataset, etc. You may want to start with 0, and test other values to see how to train faster. For me, 0 works better.

# In[ ]:


dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=0)


# ### Visualize data

# In[ ]:


dls.show_batch(sharey=True)


# ## Build learner 🏗

# In[ ]:


model = InceptionTime(dls.vars, dls.c)
learn = Learner(dls, model, metrics=accuracy)
learn.save('stage0')


# ## Train model 🚵🏼‍

# ### LR find 🔎

# In[ ]:


learn.load('stage0')
learn.lr_find()


# ### Train 🏃🏽‍♀️

# In[ ]:


learn.fit_one_cycle(25, lr_max=1e-3)
learn.save('stage1')


# In[ ]:


learn.recorder.plot_metrics()


# Let's pretend we need to end the working session now for some reason, but we'd like to continue working with this datasets and learner in the future. 
# 
# To save everything you can use a convenience function I've created that saves the learner with the model, the data and the opt function status: 

# In[ ]:


learn.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')


# As soon as we've done this, we can end the session, and continue at any time in the future. 
# 
# Let's simulate that we need to end the session now:

# In[ ]:


del learn, dsets, dls


# Next time we go back to work, we'll need to reload the datasets and learner (with the same status we had):

# In[ ]:


learn = load_learner_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')
dls = learn.dls
valid_dl = dls.valid
b = next(iter(valid_dl))
b


# In[ ]:


valid_probas, valid_targets, valid_preds = learn.get_preds(dl=valid_dl, with_decoded=True)
valid_probas, valid_targets, valid_preds


# We can confirm the learner has the same status it had at the end of training, by confirming the validation accuracy is the same:

# In[ ]:


(valid_targets == valid_preds).float().mean()


# Great! It's the same. This means we have now the learner at the same point where we left it.

# ## Visualize results 👁

# In[ ]:


learn.show_results()


# In[ ]:


learn.show_probas()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[ ]:


interp.most_confused(min_val=3)


# ## Inference on additional data 🆕

# Let's say we want to predict labels on new data. Let's see how this works.

# We may have additional data (test set) where we want to check our performance. In this case, we'd add a labelled dataset:

# In[ ]:


# Labelled test data
test_ds = valid_dl.dataset.add_test(X, y)# In this case I'll use X and y, but this would be your test data
test_dl = valid_dl.new(test_ds)
next(iter(test_dl))


# By selecting the valid dataset (valid_dl.dataset) we ensure that the same tfms applied to the valid data will be applied to the new data.

# In[ ]:


test_probas, test_targets, test_preds = learn.get_preds(dl=test_dl, with_decoded=True, save_preds=None, save_targs=None)
test_probas, test_targets, test_preds


# In[ ]:


print(f'accuracy: {skm.accuracy_score(test_targets, test_preds):10.6f}')


# If data is unlabelled, we'd just do this: 

# In[ ]:


# Unlabelled data
test_ds = dls.dataset.add_test(X)
test_dl = valid_dl.new(test_ds)
next(iter(test_dl))


# In[ ]:


test_probas, *_ = learn.get_preds(dl=test_dl, save_preds=None)
test_probas


# ## Summary ✅

# This is all the code you need to train a TS model. As you can see, it's v2 is easier to use and faster compared to v1.

# In[ ]:


dsid = 'NATOPS' 
X, y, splits = get_UCR_data(dsid, return_split=False)
tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=0)
model = InceptionTime(dls.vars, dls.c)
learn = Learner(dls, model, metrics=accuracy)
learn.fit_one_cycle(25, lr_max=1e-3)
learn.plot_metrics()


# ## New scikit-learn-like API 🎉

# As of `tsai` version 0.2.15 I have added a new scikit-learn-like API to further simplify the learner creation. 
# 
# I will prepare a new tutorial to further demonstrate how you can use the new API.
# 
# This is how you can use it for Time Series Classification: 

# In[ ]:


dsid = 'NATOPS' 
X, y, splits = get_UCR_data(dsid, return_split=False)
learn = TSClassifier(X, y, splits=splits, bs=[64, 128], batch_tfms=[TSStandardize()], arch=InceptionTime, metrics=accuracy)
learn.fit_one_cycle(25, lr_max=1e-3)
learn.plot_metrics()

