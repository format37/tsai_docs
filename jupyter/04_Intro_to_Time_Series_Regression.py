#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/04_Intro_to_Time_Series_Regression.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# created by Ignacio Oguiza - email: oguiza@timeseriesAI.co

# ## Purpose 😇

# The purpose of this notebook is to show you how you can create a simple, end-to-end, state-of-the-art **time series regression** model using **`fastai`** and **`tsai`**.
# 
# A time series regression is a task in which you assign a continuous value to a univariate or multivariate time series. 

# ## Import libraries 📚

# In[ ]:


# # **************** UNCOMMENT AND RUN THIS CELL IF YOU NEED TO INSTALL/ UPGRADE TSAI ****************
# stable = True # Set to True for latest pip version or False for main branch in GitHub
# !pip install {"tsai -U" if stable else "git+https://github.com/timeseriesAI/tsai.git"} >> /dev/null


# In[ ]:


from tsai.all import *
my_setup()


# ## Prepare data 🔢

# We are going to select a dataset from the recently released Monash, UEA & UCR 
# Time Series Extrinsic Regression Repository (2020) ([web](http://tseregression.org), [paper](https://arxiv.org/abs/2006.10996)). 
# 
# Please, feel free to select any other dataset to experiment with it. Here's the entire list.

# In[ ]:


regression_list


# In[ ]:


dsid = 'AppliancesEnergy' 
X, y, splits = get_regression_data(dsid, split_data=False)
X.shape, y.shape, y[:10]


# For regression tasks, we need to ensure y is a float. Let's check the format of the data:

# In[ ]:


check_data(X, y, splits)


# In[ ]:


tfms  = [None, [TSRegression()]]
batch_tfms = TSStandardize(by_sample=True, by_var=True)
dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=128)
dls.one_batch()


# `TSDatasets` identifies this as a regression problem, as the 2nd output (the ys) are floats. That's why the number of classes is set to 1. This is required to be able to correctly use the time series models available in `timesereisAI`.

# In[ ]:


dls.c


# In[ ]:


dls.show_batch()


# ## Build learner 🏗

# * Model: we can choose any of the time series models available in `timeseriesAI`. The same ones that work for classification also work for regression. In this case we'll use a state-of-the-art time series model called ` InceptionTime`.
# * Loss: since this is a regression problem, we''l use a regression loss (`MSELossFlat`). However, there's not need to pass it to the Learner, as it will automatically infer the required loss.
# * Metrics: we'll also choose regression metrics. (`mse` will return the same result as the loss we have selected. Just added it for demo purposes).

# In[ ]:


learn = ts_learner(dls, InceptionTime, metrics=[mae, rmse], cbs=ShowGraph())
learn.lr_find()


# It seems we can use a lr around 1e-2. Let's try it.

# In[ ]:


learn.loss_func


# ## Train model 🚵🏼‍

# In[ ]:


learn = ts_learner(dls, InceptionTime, metrics=[mae, rmse], cbs=ShowGraph())
learn.fit_one_cycle(50, 1e-2)


# In[ ]:


PATH = Path('./models/Regression.pkl')
PATH.parent.mkdir(parents=True, exist_ok=True)
learn.export(PATH)


# In[ ]:


del learn


# ## Inference ⎘

# We'll now upload the saved learner and create the predictions:

# In[ ]:


PATH = Path('./models/Regression.pkl')
learn = load_learner(PATH, cpu=False)


# In[ ]:


probas, _, preds = learn.get_X_preds(X[splits[1]])
skm.mean_squared_error(y[splits[1]], preds, squared=False)


# As you can see, this matches the valid rmse at the end of training, so the model is predicting correctly. Now you can pass any data and generate other predictions.

# ## Summary ✅

# As you can see, to use fastai and timeseriesAI to perform a time series regression/ forecasting task is pretty easy. The only thing you need to make sure is that:
# 
# * Your data is correctly prepared (with ys as floats)
# * Select the right metrics (Learner will automatically select the right loss, unless you want to pass a specific one yourself).
