#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/00b_How_to_use_numpy_arrays_in_fastai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# created by Ignacio Oguiza - email: oguiza@timeseriesAI.co

# ## How to work with numpy arrays in fastai: a time series classification example ⏳ 

# I'd like to share with you how you can work with numpy arrays in **fastai** through a time series classification example. 
# 
# I've used timeseriesAI (based on v1 extensively). To be able to use fastai v2 I have a few requirements: 
# 
# * Use univariate and multivariate time series
# * Use labelled (X,y) and unlabelled (X,) datasets
# * Data may be already split in train/valid
# * In-memory and on-disk np.arrays (np.memmap in case of larger than RAM data)
# * Slice the dataset (based on selected variables and/ or sequence steps)
# * Use item and batch tfms (transforms)
# * Create batch with specified output types (TSTensor, TensorCategory, etc)
# * Show batch (with tfms)
# * Show results
# * Add test data and unlabelled datasets
# * Export and predict on new data
# * Equal or better performance than native Pytorch, fastai v1 & vanilla fastai v2
# 
# These are pretty challenging. Let's see if fastai can meet them (with limited customization).

# ## Import libraries 📚

# In[ ]:


# **************** UNCOMMENT AND RUN THIS CELL IF YOU NEED TO INSTALL/ UPGRADE TSAI ****************
stable = True # Set to True for latest pip version or False for main branch in GitHub
get_ipython().system('pip install {"tsai -U" if stable else "git+https://github.com/timeseriesAI/tsai.git"} >> /dev/null')


# In[ ]:


from tsai.all import *
my_setup()


# ## Load data 🔢

# In[ ]:


# dataset id
dsid = 'StarLightCurves'
X_train, y_train, X_valid, y_valid = get_UCR_data(dsid, parent_dir='./data/UCR/', verbose=True, on_disk=False)
X_on_disk, y_on_disk, splits = get_UCR_data(dsid, parent_dir='./data/UCR/', verbose=True, on_disk=True, return_split=False)
X_in_memory, y_in_memory, splits = get_UCR_data(dsid, parent_dir='./data/UCR/', verbose=True, on_disk=False, return_split=False)


# In[ ]:


bs = 128
idx = np.random.randint(len(X_in_memory), size=bs)
train_idx = np.random.randint(len(splits[0]), size=bs)
valid_idx = np.random.randint(len(splits[1]), size=bs)


# ## Building blocks: NumpyTensor/ TSTensor 🧱

# Since fastai is based on Pytorch, you'll need to somehow transform the numpy arrays to tensors (NumpyTensor or TSTensor for TS). 
# 
# There are transform functions called ToNumpyTensor/ ToTSTensor that transform an array into a tensor of type NumpyTensor/ TSTensor (both have a show method).

# In[ ]:


nt = NumpyTensor(X_in_memory)
print(nt)
nt.show();


# In[ ]:


tstensor = TSTensor(X_in_memory)
print(tstensor)
tstensor[0].show();


# ## Performance benchmarks ⏱

# In fastai v2 there are multiple options to create dataloaders. Let's see some of them and most importantly whether they meet our requirements.
# 
# I will compare performance on 2 processes: 
# 
# - cycle_dl: process to cycle through the entire valid dataset (adapted from a function developed by Thomas Capelle (fastai's @tcapelle))
# - train model for 25 epochs

# ### Pytorch dataloader & NumpyDataset

# For reference, we'll test performance in the most simple dataset we can have and the native Pytorch dataloader:

# In[ ]:


valid_ds    = TorchDataset(np.array(X_valid), np.array(y_valid).astype(int))
valid_dl    = torch.utils.data.DataLoader(valid_ds, batch_size=128)
xb, yb = next(iter(valid_dl))
xb, yb


# In[ ]:


timer.start()
cycle_dl(valid_dl)
timer.stop()


# In[ ]:


timer.start()
cycle_dl_to_device(valid_dl)
timer.stop()


# In[ ]:


get_ipython().run_line_magic('timeit', 'xb.to(default_device()), yb.to(default_device())')


# This is very fast, but:
# 
# * batch is returned on **cpu** (so additional time is required to pass it to the gpu). The challenging benchmark is the one in the cycle_dl_to_device(valid_dl) test.
# * cannot be easily integrated with fastai.
# 
# It will be difficult to find a solution that performs at the same level 😅!

# If you want a simple solution that you can use with fastai you will need to use DataLoader (fastai's default dataloader) instead of Pytorch's native dataloader.

# In[ ]:


train_ds = TSDataset(np.array(X_train), np.array(y_train).astype(int) - 1, types=(TSTensor, TSLabelTensor))
train_dl = DataLoader(train_ds, bs=128, num_workers=0)
valid_ds = TSDataset(np.array(X_valid), np.array(y_valid).astype(int) - 1, types=(TSTensor, TSLabelTensor))
valid_dl = DataLoader(valid_ds, bs=128, num_workers=0)
dls      = DataLoaders(train_dl, valid_dl, device=default_device())
xb,yb = next(iter(dls.valid))
print(xb, yb)
print(f'shape: {str(len(train_ds)):10}   bs: {xb.shape}')
timer.start()
cycle_dl(dls.valid)
timer.stop()


# But this is relatively slow, as DataLoader processes samples one at a time, and each item needs to be cast to the appropriate type.
# 
# Note: we'll see how this can be accelerated later by processing all batch samples at once.

# In[ ]:


c = len(np.unique(y_train))
model = InceptionTime(X_train.shape[-2], c)
learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)
timer.start()
learn.fit_one_cycle(25, lr_max=1e-3)
timer.stop()


# ### Fastai v1

# For comparison, I've run the same exact test in the same machine with fastai v1 timeseries code and these are the timings: : 
# 
# - cycle_dl:  1.01s
# - training time: 102 s
# 
# These are the timings we'd like to beat if we want to have a faster TS framework.

# ### Fastai v2:  Factory method

# Since UCR data was already split into train and test, we'll pass IndexSplitter(splits[1]) as splitter so we get exactly the same split.

# In[ ]:


dls = TSDataLoaders.from_numpy(X_in_memory, y_in_memory, splitter=IndexSplitter(splits[1]), bs=64, val_bs=128, num_workers=0)
next(iter(dls.valid))


# In[ ]:


get_ipython().run_line_magic('time', 'cycle_dl(dls.train)')


# In[ ]:


get_ipython().run_line_magic('time', 'cycle_dl(dls.valid)')


# In[ ]:


model = InceptionTime(X_in_memory.shape[-2], dls.c)
learn = Learner(dls, model, metrics=accuracy)
timer.start()
learn.fit_one_cycle(25, lr_max=1e-3)
timer.stop()


# This method is very easy to use, but it's pretty slow.

# ### Fastai v2:  Datablock API

# In[ ]:


getters = [ItemGetter(0), ItemGetter(1)]
dblock = DataBlock(blocks=(TSTensorBlock, CategoryBlock()),
                   getters=getters,
                   splitter=IndexSplitter(splits[1]),
                   item_tfms=None,
                   batch_tfms=None)
source = itemify(X_in_memory, y_in_memory)
dls = dblock.dataloaders(source, bs=64, val_bs=128, num_workers=0)
xb,yb = next(iter(dls.valid))
print(f'shape: {str(len(dls.valid.dataset)):10}   bs: {xb.shape}')


# In[ ]:


timer.start()
cycle_dl(dls.train)
timer.stop()


# In[ ]:


timer.start()
cycle_dl(dls.valid)
timer.stop()


# So it takes more than 3 seconds to cycle the entire dataloader. This is much slower than Pytorch simple model (although fastai v2 provides a lot more functionality!).

# In[ ]:


model = InceptionTime(X_in_memory.shape[-2], dls.c)
learn = Learner(dls, model, metrics=accuracy)
timer.start()
learn.fit_one_cycle(25, lr_max=1e-3)
timer.stop()


# This is very slow compared to the native Pytorch, and even to fastai v1.

# ### Hybrid: Pytorch dataset + Fastai DataLoaders

# 
# 
# Sylvain Gugger provided an alternative recommendation to use numpy arrays in this [post](https://forums.fast.ai/t/datablock-with-numpy-input/64848/2):
# 
# "You can create a DataLoaders object from regular PyTorch datasets (though all the visualization methods like show_batch and show_results will fail)."

# In[ ]:


train_ds = TorchDataset(np.array(X_train), np.array(y_train).astype(int) - 1)
valid_ds = TorchDataset(np.array(X_valid), np.array(y_valid).astype(int) - 1)
dls = DataLoaders.from_dsets(train_ds, valid_ds, batch_size=128, num_workers=0, device=default_device())
xb,yb = next(iter(dls.valid))
print(f'shape: {str(len(dls.valid.dataset)):10}   bs: {xb.shape}')


# In[ ]:


timer.start()
cycle_dl(dls.train)
timer.stop()


# In[ ]:


timer.start()
cycle_dl(dls.valid)
timer.stop()


# In[ ]:


model = InceptionTime(X_in_memory.shape[-2], len(np.unique(y_in_memory)))
learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)
timer.start()
learn.fit_one_cycle(25, lr_max=1e-3)
timer.stop()


# This is definitely an improvement in terms of speed.
# 
# It is now better than fastai v1! 

# ## NumpyDatasets & NumpyDataLoaders 🤩

# So far we we've seen fastai v2 is very flexible and easy to use, but it's slow compared to v1 (in this example the Datablock API was 65% slower). 
# 
# There are at least 3 major differences between vision and time series -TS- (and numpy based data in general) that we can leverage to improve performance: 
# 
# 1. Vision typically requires some item preprocessing that is sometimes random. For example, when you randomly crop an image. Each time it'll return a different value. However, with time series, most item transforms are **deterministic** (actually most impact the label only).
# 
# 2. In vision problems, you usually derive the image and label from a single item (path). In TS problems, it's common to have data already **split between X and y**. It doesn't make much sense to have data already split (into X and y) to merge them in a single item and process them together. 
# 
# 3. In vision problems, you can only create a batch processing one image at a time. However with numpy datasets, you can create a batch **processing all batch items** at the same time, just by slicing an array/ tensor, which is much faster.
# 
# Based on these ideas, we could modify datasets and dataloader and:
# 
# 1. **Preprocess item tfms in place** during datasets initialization, and thus save this time in every epoch
# 
# 2. **Apply the tfms independently to the inputs (X) and labels (y)**. The output of this process 2 arrays or tensors that can be easily sliced. Slicing is a much faster operation than applying a transform.
# 
# 3. Remove the collate function, and instead **slice using all indices at the same time**. Then we can cast the output to the desired subclasses.
# 
# To test this approach, I've created a NumpyDatasets and NumpyDataLoader that leverage the characteristics of numpy-based datasets.
# 
# BTW, something important as well, is that fastai v2 design allows the use of larger than RAM datasets, as data can be sliced directly from disk before loading in memory. If you want to learn more about the usage of np.memmap you may want to see nb 00.

# You can use inplace=True whenever you want to speed up training **and**:
# 
# * you are **not using any tfms, or**
# * if you are using tfms, **transformed X (and y) both fit in memory**
# 
# Using `inplace` won't be effective for items of type np.memmap (to avoid trying to load in memory larger that RAM datasets). In many time series problems, X transforms can be applied after a batch has been created. In all these cases, you can use inplace=True.
# 
# `inplace=true` only impact item transforms.

# In[ ]:


tfms  = [None, [Categorize()]]
dsets = TSDatasets(X_in_memory, y_in_memory, tfms=tfms, splits=splits, inplace=True)
print(dsets[0])
show_at(dsets, 0);


# In[ ]:


tfms  = [None, [Categorize()]]
dsets = TSDatasets(X_in_memory, y_in_memory, tfms=tfms, splits=splits, inplace=True)
print(dsets[0])
show_at(dsets, 0);


# If you have test data, you can just do this: 

# In[ ]:


test_ds = dsets.add_test(X_in_memory, y_in_memory)
test_ds[0]


# To create dataloaders, you just need this:

# In[ ]:


dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], num_workers=0)
b = next(iter(dls.train))
b


# In[ ]:


dls.train.show_batch(sharey=True)


# Let's now establish another benchmark. 
# 
# If we think of it, the fastest and simplest way to create a batch to be used in fastai v2 would be to:
# 
# 1. split the X and y between train and valid. This can be done at initialization. 
# 
# 2. Slice the data based on random idx, cast the outputs to the expected classes, and create a tuple. 
# 
# This process takes about 200 µs in my machine. So it's very fast.

# In[ ]:


X_val = X_in_memory[splits[1]]
y_val = y_in_memory[splits[1]].astype(int)
tuple((TSTensor(X_val[valid_idx]), TensorCategory(y_val[valid_idx])))


# In[ ]:


get_ipython().run_line_magic('timeit', 'tuple((TSTensor(X_val[valid_idx]), TensorCategory(y_val[valid_idx])))')


# Let's see how this compares to NumpyDatasets when tfms are not preprocessed:

# In[ ]:


# Preprocess = False
tfms  = [None, [Categorize()]]
dsets = TSDatasets(X_in_memory, y_in_memory, sel_vars=None, sel_steps=None, tfms=tfms, splits=splits, inplace=False)
dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], num_workers=0)


# In[ ]:


valid_ds = dsets.valid
get_ipython().run_line_magic('timeit', 'valid_ds[valid_idx]')


# In[ ]:


timer.start()
cycle_dl(dls.train)
timer.stop()


# In[ ]:


timer.start()
cycle_dl(dls.valid)
timer.stop()


# Let's see how the performance when data is preprocessed:

# In[ ]:


# Preprocess = True
tfms  = [None, [Categorize()]]
dsets = TSDatasets(X_in_memory, y_in_memory, sel_vars=None, sel_steps=None, tfms=tfms, splits=splits, inplace=True)
dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], num_workers=0)


# In[ ]:


valid_ds = dsets.valid
get_ipython().run_line_magic('timeit', 'valid_ds[valid_idx]')


# 😲 Wow! This is superfast! Since we only perform slicing and casting at batch creation time performance is excellent. And it's much faster than when inplace=False. 

# In[ ]:


timer.start()
cycle_dl(dls.train)
timer.stop()


# In[ ]:


timer.start()
cycle_dl(dls.valid)
timer.stop()


# 🙃 This is even faster than the simple Pytorch dataloader, and much more flexible and with many additional benefits ❣️

# Let's now measure the timing with data on-disk instead of in memory.

# In[ ]:


# Preprocess = True
tfms  = [None, [Categorize()]]
dsets = TSDatasets(X_on_disk, y_on_disk, sel_vars=None, sel_steps=None, tfms=tfms, splits=splits, inplace=True)
dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], num_workers=0)


# In[ ]:


valid_ds = dsets.valid
get_ipython().run_line_magic('timeit', 'valid_ds[valid_idx]')


# In[ ]:


timer.start()
cycle_dl(dls.train)
timer.stop()


# In[ ]:


timer.start()
cycle_dl(dls.valid)
timer.stop()


# ⚠️ There's a delay in batch creation when data is on disk, but it's not too bad. It shouldn't have much impact during training!

# 
# 
# Let's now compare the time to train the model.

# In[ ]:


# inplace=False, Data in memory
tfms  = [None, [Categorize()]]
dsets = TSDatasets(X_in_memory, y_in_memory, tfms=tfms, splits=splits, inplace=False)
dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], num_workers=0)
model = InceptionTime(dls.vars, dls.c)
learn = Learner(dls, model, metrics=accuracy)
timer.start()
learn.fit_one_cycle(25, lr_max=1e-3)
timer.stop()


# ⚠️ This NumpyDataLoader is faster than fastai v1 even when inplace=False. 

# In[ ]:


# inplace=True, Data in memory
tfms  = [None, [Categorize()]]
dsets = TSDatasets(X_in_memory, y_in_memory, tfms=tfms, splits=splits, inplace=True)
dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], num_workers=0)
model = InceptionTime(dls.vars, dls.c)
learn = Learner(dls, model, metrics=accuracy)
timer.start()
learn.fit_one_cycle(25, lr_max=1e-3)
timer.stop()


# 🍻 🎉 I think this is a great result. It means that just preprocessing the item transforms can greatly reduce total training time!!

# In[ ]:


# inplace=True, Data on disk
tfms  = [None, [Categorize()]]
dsets = TSDatasets(X_on_disk, y_on_disk, tfms=tfms, splits=splits, inplace=True)
dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], num_workers=0)
model = InceptionTime(dls.vars, dls.c)
learn = Learner(dls, model, metrics=accuracy)
timer.start()
learn.fit_one_cycle(25, lr_max=1e-3)
timer.stop()


# ⚠️ This is also very important, as it means we can now train very large datasets with a good performance without loading data in memory.

# ## End-to-end process with recommended approach 🏁

# Let's simulate an end-to-end process to confirm everything works as expected.
# 
# We'll first build the datasets, learner and train a model:

# In[ ]:


dsid = 'NATOPS'
X, y, splits = get_UCR_data(dsid, parent_dir='./data/UCR/', verbose=True, on_disk=True, return_split=False)


# In[ ]:


tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits) # inplace=True by default
dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=0)
model = InceptionTime(dls.vars, dls.c)
learn = Learner(dls, model, metrics=accuracy)
learn.fit_one_cycle(25, lr_max=1e-3)
learn.recorder.plot_metrics()


# Let's simulate we need to end the working session now but want to continue working with this datasets and learner in the future. 
# 
# To save everything you can use a convenience function I've created that saves the learner with the model, the data and the opt function status: 

# In[ ]:


learn.save_all()


# As soon as we've done this, we can end the session, and continue at any time in the future. 
# 
# Let's simulate that we need to end the session now:

# In[ ]:


del learn, dsets, dls


# Next time we go back to work, we'll need to reload the datasets and learner (with the same status we had):

# In[ ]:


learn = load_learner_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner', device='cpu')
dls = learn.dls
first(dls.valid)


# We can now analyze the results:

# In[ ]:


learn.show_results(sharey=True)


# In[ ]:


valid_probas, valid_targets, valid_preds = learn.get_preds(dl=dls.valid, with_decoded=True)
valid_probas, valid_targets, valid_preds


# We can confirm the learner has the same status it had at the end of training, by confirming the validation accuracy is the same:

# In[ ]:


(valid_targets == valid_preds).float().mean()


# Great! It's the same. This means we have now the learner at the same point where we left it.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# ### Add additional labelled test data

# In[ ]:


# labelled test data
test_ds = dls.valid.dataset.add_test(X, y)
test_dl = dls.valid.new(test_ds)
next(iter(test_dl))


# In[ ]:


test_probas, test_targets, test_preds = learn.get_preds(dl=test_dl, with_decoded=True, save_preds=None, save_targs=None)
test_probas, test_targets, test_preds


# ### Add additional unlabelled test data

# In[ ]:


# Unlabelled test data
test_ds = dls.dataset.add_test(X)
test_dl = dls.valid.new(test_ds)
next(iter(test_dl))


# In[ ]:


test_probas, *_ = learn.get_preds(dl=test_dl, save_preds=None)
test_probas


# ## Conclusions ✅

# In summary, we've seen how we can now enjoy all the benefits of v2 when using numpy arrays with a simple scikit-learn-like API, that is much faster than v1. 
# 
# The key benefits are: 
# 
# * We can easily use numpy arrays (or anything that can be converted into np arrays). For example, this can be used for **univariate and multivariate time series**.
# * Easy to use scikit-learn type of API (X, (y))
# * We can use both **labelled and unlabelled datasets**
# * We can also use **larger than RAM datasets**, keeping data on disk (using np.memmap -see [notebook 00](00_How_to_efficiently_work_with_very_large_numpy_arrays.ipynb) for more details-).
# * Use item and batch tfms
# * Show batch method after tfms have been applied
# * Show results after training
# * **Easily export** the model to continue at a later time.
# * With NumpyDatasets + NumpyDataLoaders batch creation is **25+x faster than fastai v1** and **100+ times faster than vanilla fastai v2** (for numpy arrays).
# * This results in **2.5-3x faster training** than fastai v1 and **4-5x faster than vanilla fastai v2** (for numpy arrays).

# In[ ]:




