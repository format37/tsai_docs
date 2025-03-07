#!/usr/bin/env python
# coding: utf-8

# # Purpose 😇

# This is a brief notebook to demonstrate how you can save a learner once your model has been trained for later inference (to generate predictions) or to continue training it when new samples become available. 

# # Install and load libraries 📚

# In[ ]:


# **************** UNCOMMENT AND RUN THIS CELL IF YOU NEED TO INSTALL/ UPGRADE TSAI ****************
stable = False # Set to True for latest pip version or False for main branch in GitHub
get_ipython().system('pip install {"tsai -U" if stable else "git+https://github.com/timeseriesAI/tsai.git"} >> /dev/null')


# In[ ]:


from tsai.all import *
my_setup()


# # Train model 🏃‍♀️

# In[ ]:


X, y, splits = get_UCR_data('LSST', split_data=False)
tfms = [None, TSClassification()]
batch_tfms = TSStandardize(by_sample=True)
dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, path='/data/')
learn = ts_learner(dls, InceptionTimePlus, metrics=accuracy, cbs=[ShowGraph()])
learn.fit_one_cycle(10, 1e-2)


# # Inference: learn.export and load_learner 🚚

# If you have finished training, you can export the model for inference using the `export` method.

# In[ ]:


learn.export('exported.pth')


# When you need to generate predictions you just fo this:

# In[ ]:


new_X, *_ = get_UCR_data('LSST', split_data=False)
learn1 = load_learner('/data/exported.pth', cpu=False) # set cpu to True or False depending on your environment
preds, _, decoded_preds = learn1.get_X_preds(new_X)
preds, _, decoded_preds


# # Partial fit or fine tuning 🏋️‍♂️

# There's another way to export the learner keeping the optimizer state in case we need to keep training the model on some new data. `save` will save the model and optimizer state.

# In[ ]:


learn.save('test')


# When we have some new data, we'll create a new learner as before and load the model weights and optimizer state. Then we can fit the model on some more epochs or fine tune it. You can try both methods and see which one works best in your case.

# ## Incremental learning: 🙇🏽‍♀️

# In[ ]:


new_X, new_y, new_splits = get_UCR_data('LSST', split_data=False)
tfms = [None, TSClassification()]
batch_tfms = TSStandardize(by_sample=True)
dls2 = get_ts_dls(new_X, new_y, splits=new_splits, tfms=tfms, batch_tfms=batch_tfms, path='/data/')
learn2 = ts_learner(dls2, InceptionTimePlus, metrics=accuracy, cbs=[ShowGraph()])
learn2 = learn2.load('/data/models/test', device=device)
learn2.fit_one_cycle(1)


# ## Fine-tuning 📻

# In[ ]:


new_X, new_y, new_splits = get_UCR_data('LSST', split_data=False)
tfms = [None, TSClassification()]
batch_tfms = TSStandardize(by_sample=True)
dls3 = get_ts_dls(new_X, new_y, splits=new_splits, tfms=tfms, batch_tfms=batch_tfms, path='/data/')
learn3 = ts_learner(dls3, InceptionTimePlus, metrics=accuracy, cbs=[ShowGraph()])
learn3 = learn3.load('/data/models/test', device=device)
learn3.fine_tune(1)


# In[ ]:




