#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/15_PatchTST_a_new_transformer_for_LTSF.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# created by Ignacio Oguiza - email: oguiza@timeseriesAI.co

# # Purpose 😇

# In this notebook, we are going to learn how to use a new state-of-the-art time series transformer called **PatchTST** to create a long-term multivariate time series forecast (LTSF). PatchTST was introduced in the following paper:
# 
# * paper: Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2022). **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers**. arXiv preprint arXiv:2211.14730.
# * arxiv link: https://arxiv.org/abs/2211.14730
# * official repository: https://github.com/yuqinie98/PatchTST
# 
# The paper will be presented at the **ICLR 2023** Conference later this year. Here's the summary of the paper review by ICLR reviewers:
# 
# "The paper applies transformer to long term forecasting problems of multi-dimensional time series. The method is very simple: Take channels independently, break them into patches and predict the patches into the future using the transformer. The main advantage of this paper is that previous papers have applied transformers to this problem but it resulted in a very weak performance, being beaten by a simple linear methods. This paper found a way to apply the transformer successfully, beating the previous methods."
# 
# I'd like to thank the authors for publishing this paper, and for making their code available.
# 
# Below you can see the results of publised in the paper.

# ![download.png](attachment:download.png)

# # Install & import libraries 📚

# You'll need tsai >= 0.3.5 to be able to run this tutorial.

# In[ ]:


# # **************** UNCOMMENT AND RUN THIS CELL IF YOU NEED TO INSTALL/ UPGRADE TSAI ****************
# stable = True # Set to True for latest pip version or False for main branch in GitHub
# !pip install {"tsai -U" if stable else "git+https://github.com/timeseriesAI/tsai.git"} >> /dev/null


# In[ ]:


import sklearn
from tsai.basics import *
my_setup(sklearn)


# # Load and prepare data 🔢

# The starting point for this tutorial will be a dataframe that contains our long-term time series data. 
# 
# `tsai` allows you to easily download and prepare data from 9 popular datasets, including weather, traffic, electricity, exchange rate, ILI, and four ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2). These datasets have been extensively utilized for long-term time series forecasting benchmarking.
# 
# You can download all data from here: https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/
# 
# Here are the statistics of these datasets:

# | Datasets | Weather | Traffic | Electricity | Exchange | ILI | ETTh1 | ETTh2 | ETTm1 | ETTm2 |
# | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
# | Features | 21 | 862 | 321 | 8 | 7 | 7 | 7 | 7 | 7 |
# | Timesteps | 52696 | 17544 | 26304 | 7588 | 966 | 17420 | 17420 | 69680 | 69680 |
# 

# ## Data preparation steps

# There are 5 steps required to prepare data for a forecasting task in `tsai`:
# 
# 1. Prepare a dataframe with your data, including the variable you want to predict. 
# 2. Preprocess your data.
# 3. Define train, valid and test splits.
# 4. Scale your data using the train split. 
# 5. Apply a sliding window to prepare your input and output data.

# ### Prepare dataframe

# In this case, we are going to download the dataframe using get_long_term_forecasting_data. You can use any of the following datasets: "ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "exchange_rate", "traffic", "weather", or "ILI".
# 
# We are going to use a small dataset called ILI. ILI includes the weekly recorded **influenza-like illness (ILI)** patients data from Centers for
# Disease Control and Prevention of the United States between 2002 and 2021, which describes the ratio of patients seen with ILI and the total number of the patients.
# 
# The task is a multivariate long-term time series forecasting (LTSF), where multiple variables are predicted simultaneously for multiple time steps.

# In[ ]:


dsid = "ILI"
df_raw = get_long_term_forecasting_data(dsid)
df_raw


# ### Proprocess dataframe

# `tsai` provides some sklearn-style transforms that can be used to build a preprocessing pipeline. In this case we'll use the following transforms: 
# 
# * TSShrinkDataFrame: to save some memory and set the right dtypes.
# * TSDropDuplicates: to ensure there are no duplicate timestamps.
# * TSAddMissingTimestamps: to fill any missing timestamps. 
# * TSFillMissing: to fill any missing data (forward fill, then 0).
# 
# All these transforms can be applied to the entire dataset. In other words, they are not dependent on the training set. Other transforms will be applied later, when the training split is available.
# 
# You can read about all available transforms in the [docs](https://timeseriesai.github.io/tsai/data.preprocessing.html#sklearn-api-transforms).

# In[ ]:


datetime_col = "date"
freq = '7D'
columns = df_raw.columns[1:]
method = 'ffill'
value = 0

# pipeline
preproc_pipe = sklearn.pipeline.Pipeline([
    ('shrinker', TSShrinkDataFrame()), # shrink dataframe memory usage
    ('drop_duplicates', TSDropDuplicates(datetime_col=datetime_col)), # drop duplicate rows (if any)
    ('add_mts', TSAddMissingTimestamps(datetime_col=datetime_col, freq=freq)), # ass missing timestamps (if any)
    ('fill_missing', TSFillMissing(columns=columns, method=method, value=value)), # fill missing data (1st ffill. 2nd value=0)
    ], 
    verbose=True)
mkdir('data', exist_ok=True, parents=True)
save_object(preproc_pipe, 'data/preproc_pipe.pkl')
preproc_pipe = load_object('data/preproc_pipe.pkl')

df = preproc_pipe.fit_transform(df_raw)
df


# ### Define splits

# So we have transformed a multivariate time series with 966 time steps and 7 features (excluding the datetime) into:
# 
# * 803 input samples, with 7 features and 104 historical time steps
# * 803 input samples, with 7 features and 60 future time steps.

# It's very easy to create time forecasting splits in `tsai`. You can use as function called `get_forecasting_splits`:

# In[ ]:


fcst_history = 104 # # steps in the past
fcst_horizon = 60  # # steps in the future
valid_size   = 0.1  # int or float indicating the size of the training set
test_size    = 0.2  # int or float indicating the size of the test set

splits = get_forecasting_splits(df, fcst_history=fcst_history, fcst_horizon=fcst_horizon, datetime_col=datetime_col,
                                valid_size=valid_size, test_size=test_size)
splits


# However, in this example, we are going to apply the same splits they used in the original paper. You can use `get_forecasting splits` to use them. 

# In[ ]:


splits = get_long_term_forecasting_splits(df, fcst_history=fcst_history, fcst_horizon=fcst_horizon, dsid=dsid)
splits


# ### Scale dataframe

# Now that we have defined the splits for this particular experiment, we'll scale the data: 

# In[ ]:


columns = df.columns[1:]
train_split = splits[0]

# pipeline
exp_pipe = sklearn.pipeline.Pipeline([
    ('scaler', TSStandardScaler(columns=columns)), # standardize data using train_split
    ], 
    verbose=True)
save_object(exp_pipe, 'data/exp_pipe.pkl')
exp_pipe = load_object('data/exp_pipe.pkl')

df_scaled = exp_pipe.fit_transform(df, scaler__idxs=train_split)
df_scaled


# ### Apply a sliding window

# We'll approach the time series forecasting task as a supervised learning problem. Remember that `tsai` requires that both inputs and outputs have the following shape: 
# 
# ![text.png](attachment:text.png)

# To get those inputs and outputs we're going to use a function called `prepare_forecasting_data` that applies a sliding window along the dataframe:
# 
# ![sliding_window.png](attachment:sliding_window.png)

# To use `prepare_forecasting_data` we need to define some settings: 

# In[ ]:


x_vars = df.columns[1:]
y_vars = df.columns[1:]


# In[ ]:


X, y = prepare_forecasting_data(df, fcst_history=fcst_history, fcst_horizon=fcst_horizon, x_vars=x_vars, y_vars=y_vars)
X.shape, y.shape


# # Prepare the forecaster 🏋️‍♂️

# Now we'll instantiate the forecaster. In `tsai` there's a class called TSForecaster. We are going to use the same settings they used in the paper. 
# 
# You can find ILI specific settings here: https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/scripts/PatchTST/illness.sh
# 
# and default model settings here: https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/run_longExp.py

# In[ ]:


arch_config = dict(
    n_layers=3,  # number of encoder layers
    n_heads=4,  # number of heads
    d_model=16,  # dimension of model
    d_ff=128,  # dimension of fully connected network
    attn_dropout=0.0, # dropout applied to the attention weights
    dropout=0.3,  # dropout applied to all linear layers in the encoder except q,k&v projections
    patch_len=24,  # length of the patch applied to the time series to create patches
    stride=2,  # stride used when creating patches
    padding_patch=True,  # padding_patch
)


# In[ ]:


learn = TSForecaster(X, y, splits=splits, batch_size=16, path="models", pipelines=[preproc_pipe, exp_pipe],
                     arch="PatchTST", arch_config=arch_config, metrics=[mse, mae], cbs=ShowGraph())


# ☢️ This is **not good practice**, but all papers using these long-term forecasting datasets have published there data using drop_last=True in the validtion set. You should never use it in your practice. But if you want to try and replicate the results from the paper, you may want to uncomment the following line and set `learn.dls.valid.drop_last=True`. 

# In[ ]:


# learn.dls.valid.drop_last = True


# In[ ]:


learn.summary()


# As you can see this is a very small model, with only 57k parameters!

# # Train model 🏃🏿‍♂️

# In this case we'll use the same number of epochs and learning rate they used in the paper. 
# 
# ⚠️ Whenever you need to look for a good learning rate to train a model you can use:
# ```python
# lr_max = learn.lr_find().valley
# ```

# In[ ]:


learn = TSForecaster(X, y, splits=splits, batch_size=16, path="models", pipelines=[preproc_pipe, exp_pipe],
                     arch="PatchTST", arch_config=arch_config, metrics=[mse, mae], cbs=[ShowGraph()])

n_epochs = 100
lr_max = 0.0025
learn.fit_one_cycle(n_epochs, lr_max=lr_max)
learn.export('patchTST.pt')


# # Evaluate model 🕵️‍♀️

# ## Valid split

# First we are going to check that the valid predictions match the results we got during training. But you can skip this step since it's not required.

# In[ ]:


from tsai.inference import load_learner
from sklearn.metrics import mean_squared_error, mean_absolute_error

learn = load_learner('models/patchTST.pt')
scaled_preds, *_ = learn.get_X_preds(X[splits[1]])
scaled_preds = to_np(scaled_preds)
print(f"scaled_preds.shape: {scaled_preds.shape}")

scaled_y_true = y[splits[1]]
results_df = pd.DataFrame(columns=["mse", "mae"])
results_df.loc["valid", "mse"] = mean_squared_error(scaled_y_true.flatten(), scaled_preds.flatten())
results_df.loc["valid", "mae"] = mean_absolute_error(scaled_y_true.flatten(), scaled_preds.flatten())
results_df


# ## Test split

# So now we'll use the test split to measure performance (this is the one you that is published in the paper). 
# 
# ⚠️ You may find some differences due to randomness of the process. In addition to that, the authors used a test dataloader that drop the last batch if incomplete, which means that not all samples are used to measure performance. In `tsai` we are using all samples.

# In[ ]:


from tsai.inference import load_learner
from sklearn.metrics import mean_squared_error, mean_absolute_error

learn = load_learner('models/patchTST.pt')
y_test_preds, *_ = learn.get_X_preds(X[splits[2]])
y_test_preds = to_np(y_test_preds)
print(f"y_test_preds.shape: {y_test_preds.shape}")

y_test = y[splits[2]]
results_df = pd.DataFrame(columns=["mse", "mae"])
results_df.loc["test", "mse"] = mean_squared_error(y_test.flatten(), y_test_preds.flatten())
results_df.loc["test", "mae"] = mean_absolute_error(y_test.flatten(), y_test_preds.flatten())
results_df


# ### Visualize predictions

# In[ ]:


X_test = X[splits[2]]
y_test = y[splits[2]]
plot_forecast(X_test, y_test, y_test_preds, sel_vars=True)


# # Inference 🚀 

# ## Prepare dataframe

# If you want to use the model with new data, you'll need to first prepare the data following the process we defined before. 
# 
# Let's assume we want to create a prediction for '2020-06-30'. In our case, we need a history of 104 time steps to predict the next 60 days. We'll prepare data in the following way: 

# In[ ]:


fcst_date = "2020-06-30"

dates = pd.date_range(start=None, end=fcst_date, periods=fcst_history, freq=freq)
dates


# In[ ]:


new_df = get_long_term_forecasting_data(dsid, return_df=True)
new_df = new_df[new_df[datetime_col].isin(dates)].reset_index(drop=True)
new_df


# ## Preprocess dataframe

# In[ ]:


from tsai.inference import load_learner

learn = load_learner('models/patchTST.pt')
new_df = learn.transform(new_df)
new_df


# ## Apply sliding window

# In[ ]:


x_feat = new_df.columns[1:]
new_X, _ = prepare_forecasting_data(new_df, fcst_history=fcst_history, fcst_horizon=0, x_vars=x_vars, y_vars=None)
new_X.shape


# ## Cast predictions

# In[ ]:


new_scaled_preds, *_ = learn.get_X_preds(new_X)

new_scaled_preds = to_np(new_scaled_preds).swapaxes(1,2).reshape(-1, len(y_vars))
dates = pd.date_range(start=fcst_date, periods=fcst_horizon + 1, freq='7D')[1:]
preds_df = pd.DataFrame(dates, columns=[datetime_col])
preds_df.loc[:, y_vars] = new_scaled_preds
preds_df = learn.inverse_transform(preds_df)
preds_df


# # Conclusion ✅

# In this notebook we have covered the following topics:
# 
# * PatchTST: a new state-of-the-art transformer for long-term multivariate time series forecasting.
# * how to prepare data for a time series task.
# * how to use PatchTST within the tsai framework.
# * how to use predict multiple variables and multiple steps into the future.
# * how to visualize predictions and compare them to true values.
# 
# I hope you've found this helpful. Now it's your opportunity to start creating your own forecasts!

# In[ ]:




