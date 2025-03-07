#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/06_TS_to_image_classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# created by Ignacio Oguiza - email: oguiza@timeseriesAI.co

# ## Introduction 🤝

# Sometimes it may be useful to transform a univariate or multivariate time series into an image so that any of the techniques available for images can be used. 
# 
# These images can be created "on the fly" by transforms that work in the same way as any other transforms in fastai. I've added a few of these transforms to the tsai library. Most of the transforms come from the excellent `pyts` library (for more information please visit https://pyts.readthedocs.io).
# 
# I'd like to warn you up from that the transform of a TS to an image "on the fly" is slow, and can make you training way too long. If you are still interested in using TS as images, you may test wich ones work best on a small subset, and then create and save the output as images. You can then use a regular vision dataloader to train a vision model. 

# ## Import libraries 📚

# In[ ]:


# # **************** UNCOMMENT AND RUN THIS CELL IF YOU NEED TO INSTALL/ UPGRADE TSAI ****************
# stable = True # Set to True for latest pip version or False for main branch in GitHub
# !pip install {"tsai -U" if stable else "git+https://github.com/timeseriesAI/tsai.git"} >> /dev/null


# In[ ]:


from tsai.all import *
my_setup()


# ## Available TS to Image transforms ⏳

# The following time series to image transforms are available in the tsai library:
# 
# * **TSToPlot**: creates a matplotlib line plot
# * **TSToMat**: creates a matplotlib imshow plot
# * **TSToGADF**: creates an image based on a Gramian Angular Difference Filed transformation
# * **TSToGASF**: creates an image based on a Gramian Angular Summation Filed transformation
# * **TSToMTF**: creates an image based on a Markov Transition Field transformation
# * **TSToRP**: creates an image based on a Recurrence Plot transformation
# 
# All transforms can be used with **univariate or multivariate time series**.

# In[ ]:


dsid = 'NATOPS' # multivariate dataset
X, y, splits = get_UCR_data(dsid, return_split=False)
tfms = [None, Categorize()]
bts = [[TSNormalize(), TSToPlot()], 
       [TSNormalize(), TSToMat(cmap='viridis')],
       [TSNormalize(), TSToGADF(cmap='spring')],
       [TSNormalize(), TSToGASF(cmap='summer')],
       [TSNormalize(), TSToMTF(cmap='autumn')],
       [TSNormalize(), TSToRP(cmap='winter')]]
btns = ['Plot', 'Mat', 'GADF', 'GASF', 'MTF', 'RP']
for i, (bt, btn) in enumerate(zip(bts, btns)):
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
    dls = TSDataLoaders.from_dsets(dsets.train,
                                   dsets.valid,
                                   bs=[64, 128],
                                   batch_tfms=bt,
                                   shuffle=False)
    xb, yb = dls.train.one_batch()
    print(f'\n\ntfm: TSTo{btn} - batch shape: {xb.shape}')
    xb[0].show()
    plt.show()


# ## Univariate time series 🦄

# Let's first see how all these transforms can be applied to univariate time series. 

# In[ ]:


dsid = 'OliveOil'
X, y, splits = get_UCR_data(dsid, return_split=False)
epochs = 200


# ### Raw data (InceptionTime)

# As a benchmark, we'll first train a state-of-the-art TS model (Inceptiontime) that uses raw data. 

# In[ ]:


tfms = [None, Categorize()]
batch_tfms = [TSStandardize()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
dls.show_batch()


# Even if these look like images, the actual data passed to the model is the raw data (batch_size x channels x sequence lentgh). 
# 
# This is different from the TS to image transforms that will see in a moment, in which the batch contains actual images. 

# In[ ]:


model = create_model(InceptionTime, dls=dls)
learn = Learner(dls, model, metrics=accuracy, cbs=ShowGraph())
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")


# ### TSToPlot

# All of these transforms are applied as batch transforms. We can use the same TSDatasets and TSDataLoaders we always use.
# 
# Some of these transforms require a previous normalization (TSToMat, TSToGASF, TSToGADF, TSToRP). In general, I'd recommend you to use TSNormalize in all cases. By default this transform will normalize each sample between (-1, 1), although you can also choose the option by_sample=False, in which case, it will normalize based on the entire training set. 
# 
# We'll train a vision model. In our case, we'll use xresnet34, which is part of the fastai library.
# 
# I will train all models in the same way, without any hyperparameter optimization. 
# 
# In the case of univariate time series, you may also pass a matplotlib cmap to the transform (all except TSToPlot). Bear in mind that all these transforms will create a 1 channel image for each variable (1 in the case of univariate TS). This may be converted into a 3 channel image by applying a cmap. This will add extra time to the batch creation process but in some cased it improves performance. 

# In[ ]:


tfms = [None, Categorize()]
batch_tfms = [TSNormalize(), TSToPlot()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
dls.show_batch()


# In[ ]:


model = create_model(xresnet34, dls=dls)
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()


# ### TSToMat

# In[ ]:


tfms = [None, Categorize()]
batch_tfms = [TSNormalize(), TSToMat()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
dls.show_batch()


# One of the options you have is to use a pretrained model and just fine tune it, or train it entirely from scratch. Let's see how you could test this.

# In[ ]:


model = create_model(xresnet34, dls=dls, pretrained=True)
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()


# In[ ]:


model = create_model(xresnet34, dls=dls) # by default xresnet models are pretrained=False
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()


# In this case these 2 approaches have the same performance. We'll test this again later with a multivariate dataset. 

# ### TSToGADF

# In[ ]:


tfms = [None, Categorize()]
batch_tfms = [TSNormalize(), TSToGADF()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
dls.show_batch()


# In[ ]:


model = create_model(xresnet34, dls=dls)
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()


# ### TSToGASF

# In[ ]:


tfms = [None, Categorize()]
batch_tfms = [TSNormalize(), TSToGASF()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
dls.show_batch()


# In[ ]:


model = create_model(xresnet34, dls=dls)
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()


# ### TSToMTF

# In[ ]:


tfms = [None, Categorize()]
batch_tfms = [TSNormalize(), TSToMTF()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
dls.show_batch()


# In[ ]:


model = create_model(xresnet34, dls=dls)
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()


# ### TSToRP

# In[ ]:


tfms = [None, Categorize()]
batch_tfms = [TSNormalize(by_sample=True, range=(0,1)), TSToRP()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
dls.show_batch()


# In[ ]:


model = create_model(xresnet34, dls=dls)
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()


# ## Multivariate  time series ✖️

# In the case of multivariate time series they are used in the same way. 
# 
# The only differences are: 
# 
# * You may normalize data based on the training dataset or by sample. In either case you may also do it by variable (by_var=True) or not. 
# * You won't be able to pass a cmap as multivariate time series will create a one channel image for each of the time series variables. 

# In[ ]:


dsid = 'NATOPS'
X, y, splits = get_UCR_data(dsid, parent_dir='./data/UCR/', return_split=False)
epochs = 100


# ### Raw data (InceptionTime)

# In[ ]:


tfms = [TSStandardize(verbose=True), Categorize()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=None)
model = create_model(InceptionTime, dls=dls)
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()


# ### TSToPlot

# In[ ]:


tfms = [None, Categorize()]
batch_tfms = [TSNormalize(), TSToPlot()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
dls.show_batch()


# In[ ]:


model = create_model(xresnet34, dls=dls)
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()


# ### TSToMat

# In[ ]:


tfms = [None, Categorize()]
batch_tfms = [TSNormalize(), TSToMat()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
xb,yb = first(dls.train)
dls.show_batch()


# Let's check again if there's any difference between training a pre-trained model and training it from scratch.

# In[ ]:


model = create_model(xresnet34, dls=dls, pretrained=True)
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()


# In[ ]:


model = create_model(xresnet34, dls=dls) # not pretrained
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()


# ### TSToGADF

# In[ ]:


tfms = [None, Categorize()]
batch_tfms = [TSNormalize(), TSToGADF()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
dls.show_batch()


# In[ ]:


model = create_model(xresnet34, dls=dls)
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()


# ### TSToGASF

# In[ ]:


tfms = [None, Categorize()]
batch_tfms = [TSNormalize(), TSToGASF()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
dls.show_batch()


# In[ ]:


model = create_model(xresnet34, dls=dls)
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()


# ### TSToMTF

# In[ ]:


tfms = [None, Categorize()]
batch_tfms = [TSNormalize(), TSToMTF()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
dls.show_batch()


# In[ ]:


model = create_model(xresnet34, dls=dls)
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()


# ### TSToRP

# In[ ]:


tfms = [None, Categorize()]
batch_tfms = [TSNormalize(by_sample=True, range=(0,1)), TSToRP()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
dls.show_batch()


# In[ ]:


model = create_model(xresnet34, dls=dls)
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()


# ## Conclusion ✅

# As you have seen, these transforms are competitive if you compare them to a state-of-the-art raw TS model like InceptionTime (that take a 3d input - bs x nvars x seq_len) instead of the image models that take 4d inputs.
# 
# There are many ways in which you can transform univariate or multivariate time series into images. 
# 
# There are many options to explore all of this: 
# 
# * You can use many different TS to image transformations 
# * You can also use different cmaps
# * You can choose different image sizes
# * You can normalize data by sample and/ or by_channel
# * You can use pretrained models or train from scratch
# * You can combine TS to image transforms with other transforms
# * You can use different vision models
# 
# 
# I hope you find this useful as an introduction to this field. 

# In[ ]:




