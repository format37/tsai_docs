#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/10_Time_Series_Classification_and_Regression_with_MiniRocket.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# created by Malcolm McLean and Ignacio Oguiza (oguiza@timeseriesAI.co) based on: 
# 
# * Dempster, A., Schmidt, D. F., & Webb, G. I. (2020). MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification. arXiv preprint arXiv:2012.08791.
# 
# * Original paper: https://arxiv.org/abs/2012.08791
# 
# * Original code:  https://github.com/angus924/minirocket

# # MiniRocket 🚀
# 
# > A Very Fast (Almost) Deterministic Transform for Time Series Classification.

# ROCKET is a type of time series classification and regression methods that is different to the 
# ones you may be familiar with. Typical machine learning classifiers will 
# optimize the weights of convolutions, fully-connected, and pooling layers, 
# learning a configuration of weights that classifies the time series.
# 
# In contrast, ROCKET applies a large number of fixed, non-trainable, independent convolutions 
# to the timeseries. It then extracts a number of features from each convolution 
# output (a form of pooling), generating typically 10000 features per sample. (These 
# features are simply floating point numbers.)
# 
# The features are stored so that they can be used multiple times. 
# It then learns a simple linear head to predict each time series sample from its features. 
# Typical PyTorch heads might be based on Linear layers. When the number of training samples is small,
# sklearn's RidgeClassifier is often used.
# 
# The convolutions' fixed weights and the pooling method have been chosen experimentally to 
# effectively predict a broad range of real-world time series.
# 
# The original ROCKET method used a selection of fixed convolutions with weights 
# chosen according to a random distribution. Building upon the lessons learned 
# from ROCKET, MiniRocket refines the convolutions to a specific pre-defined set 
# that proved to be at least as effective ROCKET's. It is also much faster 
# to calculate than the original ROCKET. Actually, the paper authors "suggest that MiniRocket should now be considered and used as the default variant of Rocket."
# 
# MiniROCKET was implemented in Python using numba acceleration and mathematical 
# speedups specific to the algorithm. It runs quite fast, utilizing CPU cores in 
# parallel. Here we present a 2 implementations of MiniRocket: 
#  * a cpu version with an sklearn-like API (that can be used with small datasets - <10k samples), and
#  * a PyTorch implementation of MiniRocket, optimized for 
# the GPU. It runs faster (3-25x depending on your GPU) than the CPU version and offers some flexibility for further experimentation.
# 
# We'll demonstrate how you can use both of them througout this notebook.

# # Import libraries 📚

# In[ ]:


# # **************** UNCOMMENT AND RUN THIS CELL IF YOU NEED TO INSTALL/ UPGRADE TSAI & SKTIME ****************
# stable = False # Set to True for latest pip version or False for main branch in GitHub
# !pip install {"tsai -U" if stable else "git+https://github.com/timeseriesAI/tsai.git"} >> /dev/null
# !pip install sktime -U  >> /dev/null


# In[ ]:


from tsai.basics import *
import sktime
import sklearn
my_setup(sktime, sklearn)


# # Using MiniRocket 🚀

# * First, create the features for each timeseries sample using the MiniRocketFeatures module (MRF). 
# MRF takes a minibatch of time series samples and outputs their features. Choosing an appropriate minibatch size
# allows training sets of any size to be used without exhausting CPU or GPU memory.
# 
#     Typically, 10000 features will characterize each sample. These features are relatively
# expensive to create, but once created they are fixed and may be used as the 
# input for further training. They might be saved for example in memory or on disk.
# 
# 
# * Next, the features are sent to a linear model. The original 
# MiniRocket research used sklearn's RidgeClassifier. When the number of samples 
# goes beyond the capacity of RidgeClassifier, a deep learning "Head" can be 
# used instead to learn the classification/regression from minibatches of features.
# 
# For the following demos, we use the tsai package to handle timeseries efficiently and clearly. tsai is fully integrated with fastai, allowing fastai's training loop and other convenience to be used. To learn more about tsai, please check out the docs and tutorials at https://github.com/timeseriesAI/tsai
# 
# Let's get started.

# ## sklearn-type API (<10k samples) 🚶🏻‍♂️

# We'll first import the models we are going to use:

# In[ ]:


from tsai.models.MINIROCKET import *


# ### Classifier

# In[ ]:


# Univariate classification with sklearn-type API
dsid = 'OliveOil'
X_train, y_train, X_valid, y_valid = get_UCR_data(dsid)   # Download the UCR dataset

# Computes MiniRocket features using the original (non-PyTorch) MiniRocket code.
# It then sends them to a sklearn's RidgeClassifier (linear classifier).
model = MiniRocketClassifier()
timer.start(False)
model.fit(X_train, y_train)
t = timer.stop()
print(f'valid accuracy    : {model.score(X_valid, y_valid):.3%} time: {t}')


# In[ ]:


# Multivariate classification with sklearn-type API
dsid = 'LSST'
X_train, y_train, X_valid, y_valid = get_UCR_data(dsid)
model = MiniRocketClassifier()
timer.start(False)
model.fit(X_train, y_train)
t = timer.stop()
print(f'valid accuracy    : {model.score(X_valid, y_valid):.3%} time: {t}')


# One way to try to improve performance is to use an ensemble (that uses majority vote). Bear in mind that the ensemble will take longer since multiple models will be fitted.

# In[ ]:


# Multivariate classification ensemble with sklearn-type API
dsid = 'LSST'
X_train, y_train, X_valid, y_valid = get_UCR_data(dsid)
model = MiniRocketVotingClassifier(n_estimators=5)
timer.start(False)
model.fit(X_train, y_train)
t = timer.stop()
print(f'valid accuracy    : {model.score(X_valid, y_valid):.3%} time: {t}')


# In this case, we see an increase in accuracy although this may not be the case with other datasets.

# Once a model is trained, you can always save it for future inference: 

# In[ ]:


dsid = 'LSST'
X_train, y_train, X_valid, y_valid = get_UCR_data(dsid)
model = MiniRocketClassifier()
model.fit(X_train, y_train)
model.save(f'MiniRocket_{dsid}')
del model


# In[ ]:


model = load_minirocket(f'MiniRocket_{dsid}')
print(f'valid accuracy    : {model.score(X_valid, y_valid):.3%}')


# ### Regressor

# In[ ]:


# Univariate regression with sklearn-type API
from sklearn.metrics import mean_squared_error, make_scorer
dsid = 'Covid3Month'
X_train, y_train, X_valid, y_valid = get_Monash_regression_data(dsid)
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
model = MiniRocketRegressor(scoring=rmse_scorer)
timer.start(False)
model.fit(X_train, y_train)
t = timer.stop()
y_pred = model.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(f'valid rmse        : {rmse:.5f} time: {t}')


# In[ ]:


# Univariate regression ensemble with sklearn-type API
from sklearn.metrics import mean_squared_error, make_scorer
dsid = 'Covid3Month'
X_train, y_train, X_valid, y_valid = get_Monash_regression_data(dsid)
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
model = MiniRocketVotingRegressor(n_estimators=5, scoring=rmse_scorer)
timer.start(False)
model.fit(X_train, y_train)
t = timer.stop()
y_pred = model.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(f'valid rmse        : {rmse:.5f} time: {t}')


# In[ ]:


# Multivariate regression with sklearn-type API
from sklearn.metrics import mean_squared_error, make_scorer
dsid = 'AppliancesEnergy'
X_train, y_train, X_valid, y_valid = get_Monash_regression_data(dsid)
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
model = MiniRocketRegressor(scoring=rmse_scorer)
timer.start(False)
model.fit(X_train, y_train)
t = timer.stop()
y_pred = model.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(f'valid rmse        : {rmse:.5f} time: {t}')


# In[ ]:


# Multivariate regression ensemble with sklearn-type API
from sklearn.metrics import mean_squared_error, make_scorer
dsid = 'AppliancesEnergy'
X_train, y_train, X_valid, y_valid = get_Monash_regression_data(dsid)
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
model = MiniRocketVotingRegressor(n_estimators=5, scoring=rmse_scorer)
timer.start(False)
model.fit(X_train, y_train)
t = timer.stop()
y_pred = model.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(f'valid rmse        : {rmse:.5f} time: {t}')


# We'll also save this model for future inference:

# In[ ]:


# Multivariate regression ensemble with sklearn-type API
from sklearn.metrics import mean_squared_error, make_scorer
dsid = 'AppliancesEnergy'
X_train, y_train, X_valid, y_valid = get_Monash_regression_data(dsid)
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
model = MiniRocketVotingRegressor(n_estimators=5, scoring=rmse_scorer)
model.fit(X_train, y_train)
model.save(f'MRVRegressor_{dsid}')
del model


# In[ ]:


model = load_minirocket(f'MRVRegressor_{dsid}')
y_pred = model.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(f'valid rmse        : {rmse:.5f}')


# ## Pytorch implementation (any # samples) 🏃

# In[ ]:


from tsai.models.MINIROCKET_Pytorch import *
from tsai.models.utils import *


# ### Offline feature calculation 

# In the offline calculation, all features will be calculated in a first stage and then passed to the dataloader that will create batches. This features will ramain the same throughout training.
# 
# ⚠️ In order to avoid leakage when using the offline feature calculation, it's important to fit MiniRocketFeatures using just the train samples.

# In[ ]:


# Create the MiniRocket features and store them in memory.
dsid = 'LSST'
X, y, splits = get_UCR_data(dsid, split_data=False)


# In[ ]:


mrf = MiniRocketFeatures(X.shape[1], X.shape[2]).to(default_device())
X_train = X[splits[0]]
mrf.fit(X_train)
X_feat = get_minirocket_features(X, mrf, chunksize=1024, to_np=True)
X_feat.shape, type(X_feat)


# 👀 Note that X_train may be a np.ndarray or a torch.Tensor. In this case we'll pass a np.ndarray. 
# 
# If a torch.Tensor is passed, the model will move it to the right device (cuda) if necessary, so that it matches the model.

# We'll save this model, as we'll need it to create features in the future.

# In[ ]:


PATH = Path("./models/MRF.pt")
PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(mrf.state_dict(), PATH)


# As you can see the shape of the minirocket features is [sample_size x n_features x 1]. The last dimension (1) is added because `tsai` expects input data to have 3 dimensions, although in this case there's no longer a temporal dimension.
# 
# Once the features are calculated, we'll need to train a Pytorch model. We'll use a simple linear model:

# In[ ]:


# Using tsai/fastai, create DataLoaders for the features in X_feat.
tfms = [None, TSClassification()]
batch_tfms = TSStandardize(by_sample=True)
dls = get_ts_dls(X_feat, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
                 
# model is a linear classifier Head
model = build_ts_model(MiniRocketHead, dls=dls)
model.head


# In[ ]:


# Using tsai/fastai, create DataLoaders for the features in X_feat.
tfms = [None, TSClassification()]
batch_tfms = TSStandardize(by_sample=True)
dls = get_ts_dls(X_feat, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
                 
# model is a linear classifier Head
model = build_ts_model(MiniRocketHead, dls=dls)
                 
# Drop into fastai and use it to find a good learning rate.
learn = Learner(dls, model, metrics=accuracy, cbs=ShowGraph())
learn.lr_find()


# In[ ]:


# As above, use tsai to bring X_feat into fastai, and train.
tfms = [None, TSClassification()]
batch_tfms = TSStandardize(by_sample=True)
dls = get_ts_dls(X_feat, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
model = build_ts_model(MiniRocketHead, dls=dls)
learn = Learner(dls, model, metrics=accuracy, cbs=ShowGraph())
timer.start()
learn.fit_one_cycle(10, 3e-4)
timer.stop()


# We'll now save the learner for inference: 

# In[ ]:


PATH = Path('./models/MRL.pkl')
PATH.parent.mkdir(parents=True, exist_ok=True)
learn.export(PATH)


# #### Inference:

# For inference we'll need to follow the same process as before: 
# 
# 1. Create the features
# 2. Create predictions for those features

# Let's recreate mrf (MiniRocketFeatures) to be able to create new features: 

# In[ ]:


mrf = MiniRocketFeatures(X.shape[1], X.shape[2]).to(default_device())
PATH = Path("./models/MRF.pt")
mrf.load_state_dict(torch.load(PATH))


# We'll create new features. In this case we'll use the valid set to confirm the predictions accuracy matches the one at the end of training, but you can use any data: 

# In[ ]:


new_feat = get_minirocket_features(X[splits[1]], mrf, chunksize=1024, to_np=True)
new_feat.shape, type(new_feat)


# We'll now load the saved learner: 

# In[ ]:


PATH = Path('./models/MRL.pkl')
learn = load_learner(PATH, cpu=False)


# and pass the newly created features

# In[ ]:


probas, _, preds = learn.get_X_preds(new_feat)
preds


# In[ ]:


sklearn.metrics.accuracy_score(y[splits[1]], preds)


# Ok, so the predictions match the ones at the end of training as this accuracy is the same on we got in the end.

# ### Online feature calculation

# MiniRocket can also be used online, re-calculating the features each minibatch. In this scenario, you do not calculate fixed features one time. The online mode is a bit slower than the offline scanario, but offers more flexibility. Here are some potential uses:
# 
# * You can experiment with different scaling techniques (no standardization, standardize by sample, normalize, etc).
# * You can use data augmentation is applied to the original time series.
# * Another use of online calculation is to experiment with training the kernels and biases.
# To do this requires modifications to the MRF code.

# In[ ]:


tfms = [None, TSClassification()]
batch_tfms = TSStandardize(by_sample=True)
dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
model = build_ts_model(MiniRocket, dls=dls)
learn = Learner(dls, model, metrics=accuracy, cbs=ShowGraph())
learn.lr_find()


# Notice 2 important differences with the offline scenario: 
# 
# * in this case we pass X to the dataloader instead of X_tfm. The featurew will be calculated within the model.
# * we use MiniRocket instead of MiniRocketHead. MiniRocket is a Pytorch version that calculates features on the fly before passing them to a linear head.

# In[ ]:


tfms = [None, TSClassification()]
batch_tfms = TSStandardize(by_sample=True)
dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
model = build_ts_model(MiniRocket, dls=dls)
model


# In[ ]:


tfms = [None, TSClassification()]
batch_tfms = TSStandardize(by_sample=True)
dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
model = build_ts_model(MiniRocket, dls=dls)
learn = Learner(dls, model, metrics=accuracy, cbs=ShowGraph())
timer.start()
learn.fit_one_cycle(10, 3e-4)
timer.stop()


# Since we calculate the minirocket features within the model, we now have the option to use data augmentation for example: 

# In[ ]:


# MiniRocket with data augmentation
tfms = [None, TSClassification()]
batch_tfms = [TSStandardize(by_sample=True), TSMagScale(), TSWindowWarp()]
dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
model = build_ts_model(MiniRocket, dls=dls)
learn = Learner(dls, model, metrics=accuracy, cbs=[ShowGraph()])
learn.fit_one_cycle(20, 3e-4)


# In this case, we can see that using MiniRocket (Pytorch implementation) with data augmentation achieves an accuracy of 69%+, compared to the sklearn-API implementation which is around 65%. 

# Once you have trained the model, you can always save if for future use. We just need to export the learner:

# In[ ]:


PATH = Path('./models/MiniRocket_aug.pkl')
PATH.parent.mkdir(parents=True, exist_ok=True)
learn.export(PATH)


# In[ ]:


del learn


# #### Inference

# Let's first recreate the learner: 

# In[ ]:


PATH = Path('./models/MiniRocket_aug.pkl')
learn = load_learner(PATH, cpu=False)


# We are now ready to generate predictions. We'll confirm it works well with the valid dataset: 

# In[ ]:


probas, _, preds = learn.get_X_preds(X[splits[1]])
preds


# We can see that the validation loss & metrics are the same we had when we saved it.

# In[ ]:


sklearn.metrics.accuracy_score(y[splits[1]], preds)


# # Conclusion ✅

# MiniRocket is a new type of algorithm that is significantly faster than any other method of comparable accuracy (including Rocket), and significantly more accurate than any other method of even roughly-similar computational expense. 
# 
# `tsai` supports the 2 variations of MiniRocket introduced in this notebook. A cpu version (that can be used with relatively small datasets, with <10k samples) and a gpu (Pytorch) version that can be used with datasets of any size. The Pytorch version can be used in an offline mode (pre-calculating all features before fitting the model) or in an online mode (calculating features on the fly). 
# 
# We believe MiniRocket is a great new tool, and encourange you to try it in your next Time Series Classification or Regression task. 
