#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/00c_Time_Series_data_preparation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# created by Ignacio Oguiza - email: oguiza@timeseriesAI.co

# ## Import libraries 📚

# Since some of you have been asking questions as to how to prepare your data to be able to use timeserisAI, I've prepared a short tutorial to address this.
# 
# There are endless options in terms of how your source data may be stored, so I'll cover a few of the most frequent ones I've seen. I may be expanding this in the future if needed.

# In[ ]:


# # **************** UNCOMMENT AND RUN THIS CELL IF YOU NEED TO INSTALL/ UPGRADE TSAI ****************
# stable = True # Set to True for latest pip version or False for main branch in GitHub
# !pip install {"tsai -U" if stable else "git+https://github.com/timeseriesAI/tsai.git"} >> /dev/null


# In[ ]:


from tsai.all import *
computer_setup()


# ## Required input shape 🔶

# To be able to use timeseriesAI your data needs to have 3 dimensions: 
# 
# * **number of samples**
# * **number of features** (aka variables, dimensions, channels)
# * **number of steps** (or length, time steps, sequence steps)
# 
# There are a few convenience functions that you may want to use to prepare your data. 
# 
# We are going to see how you could prepare your data in a few scenarios. 

# **Note: I've recently modified timeseriesAI so that you can also use 2d input data in the case of univariate time series (they'll be converted to 3d internally), although you can still pass univariate time series as 3d or pass them if you prefer. You'll get the same result.**

# ## UCR time series data ⏳

# The easiest case if if you want to use some of the data already preprocessed in timeseriesAI (all UCR datasets have been included). In this case, the only thing you need to do is:
# 
# * select a univariate or multivariate dataset from the list
# * use the get_UCR_data function

# In[ ]:


print('univariate datasets: ', get_UCR_univariate_list())


# In[ ]:


print('multivariate datasets: ', get_UCR_multivariate_list())


# In[ ]:


ds_name = 'NATOPS' 
X, y, splits = get_UCR_data(ds_name, return_split=False)
X.shape, y.shape, splits


# In[ ]:


tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dsets


# As you can see, X has 3 dimensions: 
# 
# * 360 samples
# * 24 features
# * 51 time steps
# 
# Let's visualize of the samples:

# In[ ]:


plt.plot(X[0].T);


# ## 2d or 3d np.ndarray/ torch.Tensor ⌗

# Another option is that you have your data as an array or a tensor. 
# In this case, the only thing you'll need to do is to transform your data to 3d (if not already done), and generate your splits.
# We are going to simulate this scenario generating 2d data for a univariate dataset: 

# In[ ]:


ds_name = 'OliveOil' 
X, y, _ = get_UCR_data(ds_name, return_split=False)
X_2d = X[:, 0]
X_2d.shape, y.shape


# To make data 3d you use `to3d`:

# In[ ]:


X_3d = to3d(X_2d)
X_3d.shape


# To generate your splits, you would use `get_splits`. Here you need to indicate: 
# * valid_size=0.2
# * test_size (optional)
# * stratify=True if you want stratified splits
# * random_state=seed or None (random)

# In[ ]:


splits = get_splits(y, valid_size=.2, stratify=True, random_state=23, shuffle=True)
splits


# In[ ]:


X_3d.shape, y.shape, splits


# In[ ]:


tfms  = [None, [Categorize()]]
dsets = TSDatasets(X_3d, y, tfms=tfms, splits=splits, inplace=True)
dsets


# In fastai I've modified TS datasets so that you can pass univariate time series as a 2d or 3d arrays.

# In[ ]:


tfms  = [None, [Categorize()]]
dsets = TSDatasets(X_2d, y, tfms=tfms, splits=splits, inplace=True)
dsets


# ### Pre-split 2d or 3d np.ndarray/ torch.Tensor

# If your data is already split into Train and Valid/ Test, you may the use `get_predefined_split` to generate the splits:

# In[ ]:


ds_name = 'OliveOil' 
X_train, y_train, X_valid, y_valid = get_UCR_data(ds_name, return_split=True)


# In[ ]:


X, y, splits = combine_split_data([X_train, X_valid], [y_train, y_valid])


# In[ ]:


tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dsets


# ## Pandas dataframe with samples as rows 🐼

# ### Univariate

# In[ ]:


ds_name = 'OliveOil'
X, y, _ = get_UCR_data(ds_name, return_split=False)
X = X[:, 0]
y = y.reshape(-1, 1)
data = np.concatenate((X, y), axis=-1)
df = pd.DataFrame(data)
df = df.rename(columns={570: 'target'})
df.head()


# In[ ]:


X, y = df2xy(df, target_col='target')
test_eq(X.shape, (60, 1, 570))
test_eq(y.shape, (60, ))


# In[ ]:


splits = get_splits(y, valid_size=.2, stratify=True, random_state=23, shuffle=True)
splits


# In[ ]:


tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dsets


# ### Multivariate

# In[ ]:


ds_name = 'OliveOil'
X, y, _ = get_UCR_data(ds_name, return_split=False)
X = X[:, 0]
y = y.reshape(-1, 1)
data = np.concatenate((X, y), axis=-1)
df = pd.DataFrame(data).astype(float)
df = df.rename(columns={570: 'target'})
df1 = pd.concat([df, df + 10, df + 100], axis=0).reset_index(drop=False)
df2 = pd.DataFrame(np.array([1] * 60 + [2] * 60 + [3] * 60), columns=['feature'])
df = pd.merge(df2, df1, left_index=True, right_index=True)
df


# In[ ]:


X, y = df2xy(df, sample_col='index', feat_col='feature', target_col='target', data_cols=None)
test_eq(X.shape, (60, 3, 570))
test_eq(y.shape, (60, 3))


# In[ ]:


splits = get_splits(y, valid_size=.2, stratify=True, random_state=23, shuffle=True)
splits


# In[ ]:


tfms  = [None, TSRegression()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dsets


# ## Single, long time series 🤥

# Sometimes, instead of having the data already split into samples, you only have a single (univariate or multivariate) time series that you need to split. 
# The recommended way to do this is to use a sliding window. In `timeseriesAI`there is a function called `SlidingWindow`that performs this task in a flexible way.
# 
# This function applies a sliding window to a 1d or 2d input (np.ndarray, torch.Tensor or pd.DataFrame). 
#    
# * Args:
#     * window_length   = length of lookback window
#     * stride          = n datapoints the window is moved ahead along the sequence. Default: 1. If None, stride=window_length (no overlap)
#     * horizon         = number of future datapoints to predict. 0 for last step in the selected window. > 0 for future steps. List for several steps.
#     * get_x          = indices of columns that contain the independent variable (xs). If get_x=None, all data will be used as x
#     * get_y          = indices of columns that contain the target (ys). If y_idx is None, no y will be returned
#     * seq_first       = True if input shape (seq_len, n_vars), False if input shape (n_vars, seq_len)
#     * random_start    = determines the step where the first window is applied: 0 (default), a given step (int), or random within the 1st stride (None). 
# 
# * Input:
#     * shape: (seq_len, ) or (seq_len, n_vars) if seq_first=True else (n_vars, seq_len)

# ### Univariate

# You may use it just without a target

# In[ ]:


window_length = 5
t = np.arange(100)
print('input shape:', t.shape)
X, y = SlidingWindow(window_length)(t)
test_eq(X.shape, ((95, 1, 5)))


# If the target is the next step in the univariate time series set `horizon=1`:

# In[ ]:


window_length = 5
horizon = 1

t = np.arange(100)
print('input shape:', t.shape)
X, y = SlidingWindow(window_length, horizon=horizon)(t)
test_eq(X.shape, ((95, 1, 5)))
test_eq(y.shape, ((95,)))


# Horizon may be > 1 to select multiple steps in the future:

# In[ ]:


window_length = 5
horizon = 2

t = np.arange(100)
print('input shape:', t.shape)
X, y = SlidingWindow(window_length, horizon=horizon)(t)
test_eq(X.shape, ((94, 1, 5)))
test_eq(y.shape, ((94, 2)))


# To have non-overlapping samples, we need to set `stride=None`:

# In[ ]:


window_length = 5
stride = None
horizon = 1
t = np.arange(100)
print('input shape:', t.shape)
X, y = SlidingWindow(window_length, stride=stride, horizon=horizon)(t)
test_eq(X.shape, ((19, 1, 5)))
test_eq(y.shape, ((19, )))


# In[ ]:


window_length = 5
stride = 3
horizon = 1
t = np.arange(100)
print('input shape:', t.shape)
X, y = SlidingWindow(window_length, stride=stride, horizon=horizon)(t)
test_eq(X.shape, ((32, 1, 5)))
test_eq(y.shape, ((32, )))


# We can also decide where to start the sliding window using `start`: 

# In[ ]:


window_length = 5
stride = None
horizon = 1
t = np.arange(100)
print('input shape:', t.shape)
X, y = SlidingWindow(window_length, stride=stride, start=20, horizon=horizon)(t)
test_eq(X.shape, ((15, 1, 5)))
test_eq(y.shape, ((15, )))


# If the time series is of shape (1, seq_len) we need to set `seq_first=False`

# In[ ]:


window_length = 5
stride = 3
horizon = 1
t = np.arange(100).reshape(1, -1)
print('input shape:', t.shape)
X, y = SlidingWindow(window_length, stride=stride, horizon=horizon, seq_first=False)(t)
test_eq(X.shape, ((32, 1, 5)))
test_eq(y.shape, ((32, )))


# Your univariate time series may be in a pandas DataFrame:

# In[ ]:


window_length = 5
stride = None
horizon=1

t = np.arange(20)
df = pd.DataFrame(t, columns=['var'])
print('input shape:', df.shape)
display(df)
X, y = SlidingWindow(window_length, stride=stride, horizon=horizon)(df)
test_eq(X.shape, ((3, 1, 5)))
test_eq(y.shape, ((3, )))


# In[ ]:


window_length = 5
stride = None
horizon=1

t = np.arange(20)
df = pd.DataFrame(t, columns=['var']).T
print('input shape:', df.shape)
display(df)
X, y = SlidingWindow(window_length, stride=stride, horizon=horizon, seq_first=False)(df)
test_eq(X.shape, ((3, 1, 5)))
test_eq(y.shape, ((3, )))


# ### Multivariate

# When using multivariate data, all parameters shown before work in the same way, but you always need to indicate how to get the X data and the y data (as there are multiple features). To do that, we'll use get_x and get_y. 
# 
# By default get_x is set to None, which means that all features will be used.
# By default get_y is set to None, which means that all features will be used as long as horizon > 0 (to avoid leakage).
# 
# If you get the time series in a np.ndarray or a torch.Tensor, you should use integers, a list or slice as get_x/ get_y.

# In[ ]:


window_length = 5
stride = None
n_vars = 3

t = (np.random.rand(1000, n_vars) - .5).cumsum(0)
print(t.shape)
plt.plot(t)
plt.show()
X, y = SlidingWindow(window_length, stride=stride, get_x=[0,1], get_y=2)(t)
test_eq(X.shape, ((199, 2, 5)))
test_eq(y.shape, ((199, )))


# In[ ]:


window_length = 5
n_vars = 3

t = (torch.stack(n_vars * [torch.arange(10)]).T * tensor([1, 10, 100]))
df = pd.DataFrame(t, columns=[f'var_{i}' for i in range(n_vars)])
print('input shape:', df.shape)
display(df)
X, y = SlidingWindow(window_length)(df)
test_eq(X.shape, ((5, 3, 5)))
test_eq(y.shape, ((5, 3)))


# In[ ]:


window_length = 5
n_vars = 3
horizon = 1

t = (torch.stack(n_vars * [torch.arange(10)]).T * tensor([1, 10, 100]))
df = pd.DataFrame(t, columns=[f'var_{i}' for i in range(n_vars)])
print('input shape:', df.shape)
display(df)
X, y = SlidingWindow(window_length, horizon=horizon)(df)
test_eq(X.shape, ((5, 3, 5)))
test_eq(y.shape, ((5, 3)))


# You may also get the target from a different column: 

# In[ ]:


window_length = 5
n_vars = 3

t = (torch.stack(n_vars * [torch.arange(10)]).T * tensor([1, 10, 100]))
columns=[f'var_{i}' for i in range(n_vars-1)]+['target']
df = pd.DataFrame(t, columns=columns)
print('input shape:', df.shape)
display(df)
X, y = SlidingWindow(window_length, get_x=columns[:-1], get_y='target')(df)
test_eq(X.shape, ((5, 2, 5)))
test_eq(y.shape, ((5, )))


# In[ ]:


window_length = 5
n_vars = 5
horizon = 1

t = (torch.stack(n_vars * [torch.arange(10)]).T * tensor([10**i for i in range(n_vars)]))
columns=[f'var_{i}' for i in range(n_vars-1)]+['target']
df = pd.DataFrame(t, columns=columns)
print('input shape:', df.shape)
display(df)
X, y = SlidingWindow(window_length, horizon=horizon, get_x=columns[:-1], get_y='target')(df)
test_eq(X.shape, ((5, 4, 5)))
test_eq(y.shape, ((5, )))


# In[ ]:


window_length = 4
n_vars = 5
seq_len = 100
horizon = 1

t1 = (np.random.rand(seq_len, n_vars-1) - .5).cumsum(0)
t2 = np.random.randint(0, 10, (seq_len,1))
t = np.concatenate((t1, t2), axis=-1)
columns=[f'var_{i}' for i in range(n_vars-1)]+['target']
df = pd.DataFrame(t, columns=columns)
print('input shape:', df.shape)
display(df)
X, y = SlidingWindow(window_length, horizon=horizon, get_x=columns[:-1], get_y='target')(df)
splits = get_splits(y, valid_size=.2, stratify=True, random_state=23, shuffle=False)
tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dsets


# In[ ]:


dsets[0][0].data, dsets[0][1].data


# In[ ]:


window_length = 4
start = 3
n_vars = 5
seq_len = 100
horizon = 0

t1 = (np.random.rand(seq_len, n_vars-1) - .5).cumsum(0)
t2 = np.random.randint(0, 10, (seq_len,1))
t = np.concatenate((t1, t2), axis=-1)
columns=[f'var_{i}' for i in range(n_vars-1)]+['target']
df = pd.DataFrame(t, columns=columns).T
print('input shape:', df.shape)
display(df)
X, y = SlidingWindow(window_length, start=start, horizon=horizon, get_x=columns[:-1], get_y='target', seq_first=False)(df)
splits = get_splits(y, valid_size=.2, stratify=True, random_state=23, shuffle=False)
tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dsets


# In[ ]:


dsets[0][0].data, dsets[0][1].data


# ## End-to-end example 🎬

# ### Data split by sample

# This is a example using real data where the dataframe already has the data split by sample. Let's first simulate how you could get the pandas df.
# 
# In this case, you only need to convert the dataframe format to X and y using `df2xy`as we have seen before.

# In[ ]:


ds_name = 'NATOPS' 
X, y, splits = get_UCR_data(ds_name, return_split=False)
data = np.concatenate((np.arange(len(X)).repeat(X.shape[1]).reshape(-1,1), np.tile(np.arange(X.shape[1]), len(X)).reshape(-1,1)), axis=1)
df1 = pd.DataFrame(data, columns=['sample', 'feature'])
df2 = pd.DataFrame(X.reshape(-1, 51))
df3 = pd.DataFrame(np.repeat(y, X.shape[1]), columns=['target'])
df = df1.merge(df2, left_index=True, right_index=True)
df = df.merge(df3, left_index=True, right_index=True)
df


# In this case, we can shuffle the data as the individual time series are independent from the rest.

# In[ ]:


def y_func(o): return o[0]
X, y = df2xy(df, sample_col='sample', feat_col='feature', target_col='target', data_cols=df.columns[2:-1], y_func=y_func)
splits = get_splits(y, valid_size=.2, stratify=True, random_state=23, shuffle=True)
tfms  = [None, TSClassification()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dsets


# ### Single multivariate time series

# In this second scenario, we have a single time series, and we'll need to decide how to create the individual samples using the `SlidingWindow`function.
# 
# This is how you could get the dataframe, with many columns for each feature, and a target.

# In[ ]:


ds_name = 'NATOPS' 
X, y, splits = get_UCR_data(ds_name, return_split=False)
data = X.transpose(1,0,2).reshape(X.shape[1], -1)
print(X.shape, data.shape)
df = pd.DataFrame(data).T
df2 = pd.DataFrame(np.repeat(y, X.shape[2]), columns=['target'])
df = df.merge(df2, left_index=True, right_index=True)
df


# In this case, you'll need to set the following parameters:
#  
# * window_length
# * stride
# * start
# * horizon
# * get_x
# * get_y
# * seq_first
# 
# You also need to bear in mind that you sould set shuffle=False when using splits since the individual time series are correlated with rest.

# In[ ]:


window_length = X.shape[-1]  # window_length is usually selected based on prior domain knowledge or by trial and error
stride = None                # None for non-overlapping (stride = window_length) (default = 1). This depends on how often you want to predict once the model is trained
start = 0                    # use all data since the first time stamp (default = 0)
get_x = df.columns[:-1]      # Indicates which are the columns that contain the x data.
get_y = 'target'             # In multivariate time series, you must indicate which is/are the y columns
horizon = 0                  # 0 means y is taken from the last time stamp of the time sequence (default = 0)
seq_first = True
                            
X, y = SlidingWindow(window_length, stride=stride, start=start, get_x=get_x,  get_y=get_y, horizon=horizon, seq_first=seq_first)(df)
splits = get_splits(y, valid_size=.2, stratify=True, random_state=23, shuffle=False)
tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dsets


# In[ ]:


beep()

