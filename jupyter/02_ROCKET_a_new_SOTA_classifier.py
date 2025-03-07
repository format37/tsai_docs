#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/02_ROCKET_a_new_SOTA_classifier.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# created by Ignacio Oguiza - email: oguiza@timeseriesAI.co

# ## Purpose 😇

# The purpose of this notebook is to introduce you to Rocket. 
# 
# ROCKET (RandOm Convolutional KErnel Transform) is a new Time Series Classification (TSC) method that has just been released (Oct 29th, 2019), and has achieved **state-of-the-art performance on the UCR univariate time series classification datasets, surpassing HIVE-COTE (the previous state of the art since 2017) in accuracy, with exceptional speed compared to other traditional DL methods.** 
# 
# To achieve these 2 things at once is **VERY IMPRESSIVE**. ROCKET is certainly a new TSC method you should try.
# 
# Authors:
# Dempster, A., Petitjean, F., & Webb, G. I. (2019). ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels. arXiv preprint arXiv:1910.13051.
# 
# [paper](https://arxiv.org/pdf/1910.13051)
# 
# There are 2 main limitations to the original ROCKET method though:
# - Released code doesn't handle multivariate data
# - It doesn't run on a GPU, so it's slow when used with a large datasets
# 
# In this notebook you will learn: 
# - a new ROCKET version we have developed in Pytorch, that handles both **univariate and multivariate** data, and uses **GPU**
# - you will see how you can integrate the ROCKET features with fastai or other classifiers

# ## Import libraries 📚

# In[ ]:


# # **************** UNCOMMENT AND RUN THIS CELL IF YOU NEED TO INSTALL/ UPGRADE TSAI ****************
# stable = True # Set to True for latest pip version or False for main branch in GitHub
# !pip install {"tsai -U" if stable else "git+https://github.com/timeseriesAI/tsai.git"} >> /dev/null


# In[ ]:


from tsai.all import *
my_setup()


# ## How to use ROCKET on GPU?

# This method allows you to use ROCKET even with large and/or multivariate datasets on GPU in Pytorch. 

# ### 1️⃣ Generate features
# 
# First you prepare the input data and normalize it per sample. The input to ROCKET Pytorch is a 3d tensor of shape (samples, vars, len), preferrable on gpu.

# The way to use ROCKET in Pytorch is the following:
# 
# * Create a dataset as you would normally do in `tsai`. 
# * Create a TSDataLoaders with the following kwargs: 
#     * drop_last=False. In this way we get features for every input sample.
#     * shuffle_train=False
#     * batch_tfms=[TSStandardize(by_sample=True)] so that input is normalized by sample, as recommended by the authors
# 

# In[ ]:


X, y, splits = get_UCR_data('HandMovementDirection', split_data=False)
tfms  = [None, [Categorize()]]
batch_tfms = [TSStandardize(by_sample=True)]
dls = get_ts_dls(X, y, splits=splits, tfms=tfms, drop_last=False, shuffle_train=False, batch_tfms=batch_tfms, bs=10_000)


# ☣️☣️ You will be able to create a dls (TSDataLoaders) object with unusually large batch sizes. I've tested it with a large dataset and a batch size = 100_000 and it worked fine. This is because ROCKET is not a usual Deep Learning model. It just applies convolutions (kernels) one at a time to create the features.

# Instantiate a rocket model with the desired n_kernels (authors use 10_000) and kernel sizes (7, 9 and 11 in the original paper). 

# In[ ]:


model = build_ts_model(ROCKET, dls=dls) # n_kernels=10_000, kss=[7, 9, 11] set by default, but you can pass other values as kwargs


# Now generate rocket features for the entire train and valid datasets using the create_rocket_features convenience function `create_rocket_features`.

# And we now transform the original data, creating 20k features per sample

# In[ ]:


X_train, y_train = create_rocket_features(dls.train, model)
X_valid, y_valid = create_rocket_features(dls.valid, model)
X_train.shape, X_valid.shape


# ### 2️⃣ Apply a classifier

# Once you build the 20k features per sample, you can use them to train any classifier of your choice.

# #### RidgeClassifierCV

# And now you apply a classifier of your choice. 
# With RidgeClassifierCV in particular, there's no need to normalize the calculated features before passing them to the classifier, as it does it internally (if normalize is set to True as recommended by the authors).

# In[ ]:


from sklearn.linear_model import RidgeClassifierCV
ridge = RidgeClassifierCV(alphas=np.logspace(-8, 8, 17), normalize=True)
ridge.fit(X_train, y_train)
print(f'alpha: {ridge.alpha_:.2E}  train: {ridge.score(X_train, y_train):.5f}  valid: {ridge.score(X_valid, y_valid):.5f}')


# This result is amazing!! The previous state of the art (Inceptiontime) was .37837

# #### Logistic Regression

# In the case of other classifiers (like Logistic Regression), the authors recommend a per-feature normalization.

# In[ ]:


eps = 1e-6
Cs = np.logspace(-5, 5, 11)
from sklearn.linear_model import LogisticRegression
best_loss = np.inf
for i, C in enumerate(Cs):
    f_mean = X_train.mean(axis=0, keepdims=True)
    f_std = X_train.std(axis=0, keepdims=True) + eps  # epsilon to avoid dividing by 0
    X_train_tfm2 = (X_train - f_mean) / f_std
    X_valid_tfm2 = (X_valid - f_mean) / f_std
    classifier = LogisticRegression(penalty='l2', C=C, n_jobs=-1)
    classifier.fit(X_train_tfm2, y_train)
    probas = classifier.predict_proba(X_train_tfm2)
    loss = nn.CrossEntropyLoss()(torch.tensor(probas), torch.tensor(y_train)).item()
    train_score = classifier.score(X_train_tfm2, y_train)
    val_score = classifier.score(X_valid_tfm2, y_valid)
    if loss < best_loss:
        best_eps = eps
        best_C = C
        best_loss = loss
        best_train_score = train_score
        best_val_score = val_score
    print('{:2} eps: {:.2E}  C: {:.2E}  loss: {:.5f}  train_acc: {:.5f}  valid_acc: {:.5f}'.format(
        i, eps, C, loss, train_score, val_score))
print('\nBest result:')
print('eps: {:.2E}  C: {:.2E}  train_loss: {:.5f}  train_acc: {:.5f}  valid_acc: {:.5f}'.format(
        best_eps, best_C, best_loss, best_train_score, best_val_score))


# ☣️ Note: Epsilon has a large impact on the result. You can actually test several values to find the one that best fits your problem, but bear in mind you can only select C and epsilon based on train data!!! 

# ##### RandomSearch

# One way to do this would be to perform a random search using several epsilon and C values

# In[ ]:


n_tests = 10
epss = np.logspace(-8, 0, 9)
Cs = np.logspace(-5, 5, 11)

from sklearn.linear_model import LogisticRegression
best_loss = np.inf
for i in range(n_tests):
    eps = random_choice(epss)
    C = random_choice(Cs)
    f_mean = X_train.mean(axis=0, keepdims=True)
    f_std = X_train.std(axis=0, keepdims=True) + eps  # epsilon
    X_train_tfm2 = (X_train - f_mean) / f_std
    X_valid_tfm2 = (X_valid - f_mean) / f_std
    classifier = LogisticRegression(penalty='l2', C=C, n_jobs=-1)
    classifier.fit(X_train_tfm2, y_train)
    probas = classifier.predict_proba(X_train_tfm2)
    loss = nn.CrossEntropyLoss()(torch.tensor(probas), torch.tensor(y_train)).item()
    train_score = classifier.score(X_train_tfm2, y_train)
    val_score = classifier.score(X_valid_tfm2, y_valid)
    if loss < best_loss:
        best_eps = eps
        best_C = C
        best_loss = loss
        best_train_score = train_score
        best_val_score = val_score
    print('{:2}  eps: {:.2E}  C: {:.2E}  loss: {:.5f}  train_acc: {:.5f}  valid_acc: {:.5f}'.format(
        i, eps, C, loss, train_score, val_score))
print('\nBest result:')
print('eps: {:.2E}  C: {:.2E}  train_loss: {:.5f}  train_acc: {:.5f}  valid_acc: {:.5f}'.format(
        best_eps, best_C, best_loss, best_train_score, best_val_score))


# In addition to this, I have also run the code on the TSC UCR multivariate datasets (all the ones that don't contain nan values), and the results are also very good, beating the previous state-of-the-art in this category as well by a large margin. For example, ROCKET reduces InceptionTime errors by 26% on average.

# #### Fastai classifier head

# In[ ]:


X = concat(X_train, X_valid)
y = concat(y_train, y_valid)
splits = get_predefined_splits(X_train, X_valid)


# In[ ]:


tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=128, batch_tfms=[TSStandardize(by_var=True)])# per feature normalization
dls.show_batch()


# In[ ]:


def lin_zero_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.constant_(layer.weight.data, 0.)
        if layer.bias is not None: nn.init.constant_(layer.bias.data, 0.)


# In[ ]:


model = create_mlp_head(dls.vars, dls.c, dls.len)
model.apply(lin_zero_init)
learn = Learner(dls, model, metrics=accuracy, cbs=ShowGraph())
learn.fit_one_cycle(50, lr_max=1e-4)
learn.plot_metrics()


# #### XGBoost

# In[ ]:


eps = 1e-6

# normalize 'per feature'
f_mean = X_train.mean(axis=0, keepdims=True)
f_std = X_train.std(axis=0, keepdims=True) + eps
X_train_norm = (X_train - f_mean) / f_std
X_valid_norm = (X_valid - f_mean) / f_std

import xgboost as xgb
classifier = xgb.XGBClassifier(max_depth=3,
                               learning_rate=0.1,
                               n_estimators=100,
                               verbosity=1,
                               objective='binary:logistic',
                               booster='gbtree',
                               tree_method='auto',
                               n_jobs=-1,
                               gpu_id=default_device().index,
                               gamma=0,
                               min_child_weight=1,
                               max_delta_step=0,
                               subsample=.5,
                               colsample_bytree=1,
                               colsample_bylevel=1,
                               colsample_bynode=1,
                               reg_alpha=0,
                               reg_lambda=1,
                               scale_pos_weight=1,
                               base_score=0.5,
                               random_state=0,
                               missing=None)

classifier.fit(X_train_norm, y_train)
preds = classifier.predict(X_valid_norm)
(preds == y_valid).mean()


# ## Conclusions

# ROCKET is a great method for TSC that has established a new level of performance both in terms of accuracy and time. It does it by successfully applying an approach quite different from the traditional DL approaches. The method uses 10k random kernels to generate features that are then classified by linear classifiers (although you may use a classifier of your choice).
# The original method has 2 limitations (lack of multivariate and lack of GPU support) that are overcome by the Pytorch implementation shared in this notebook.
# 
# So this is all the code you need to train a state-of-the-art model using rocket and GPU in `tsai`:
# 
# ```python
# X, y, splits = get_UCR_data('HandMovementDirection', return_split=False)
# tfms  = [None, [Categorize()]]
# batch_tfms = [TSStandardize(by_sample=True)]
# dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
# dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=64, drop_last=False, shuffle_train=False, batch_tfms=[TSStandardize(by_sample=True)])
# model = create_model(ROCKET, dls=dls)
# X_train, y_train = create_rocket_features(dls.train, model)
# X_valid, y_valid = create_rocket_features(dls.valid, model)
# ridge = RidgeClassifierCV(alphas=np.logspace(-8, 8, 17), normalize=True)
# ridge.fit(X_train, y_train)
# print(f'alpha: {ridge.alpha_:.2E}  train: {ridge.score(X_train, y_train):.5f}  valid: {ridge.score(X_valid, y_valid):.5f}')
# ```
