#!/usr/bin/env python
# coding: utf-8

# created by Ignacio Oguiza - email: oguiza@timeseriesAI.co

# ## How to efficiently work with (very large) Numpy Arrays? 👷‍♀️

# Sometimes we need to work with some very large numpy arrays that don't fit in memory. I'd like to share with you a way that works well for me.

# ## Import libraries 📚

# In[ ]:


# # **************** UNCOMMENT AND RUN THIS CELL IF YOU NEED TO INSTALL/ UPGRADE TSAI ****************
# stable = True # Set to True for latest pip version or False for main branch in GitHub
# !pip install {"tsai -U" if stable else "git+https://github.com/timeseriesAI/tsai.git"} >> /dev/null


# In[ ]:


from tsai.all import *
my_setup()


# ## Introduction 🤝

# I normally work with time series data. I made the decision to use numpy arrays to store my data since they can easily handle multiple dimensions, and are really very efficient.
# 
# But sometimes datasets are really big (many GBs) and don't fit in memory. So I started looking around and found something that works very well: [**np.memmap**](https://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html). Conceptually they work as arrays on disk, and that's how I often call them.
# 
# np.memmap creates a map to numpy arrays you have previously saved on disk, so that you can efficiently access small segments of those (small or large) files on disk, without reading the entire file into memory. And that's exactly what we need with deep learning, be able to quickly create a batch in memory, without reading the entire file (that is stored on disk). 
# 
# The best analogy I've found are image files. You may have a very large dataset on disk (that far exceeds your RAM). In order to create your DL datasets, what you pass are the paths to each individual file, so that you can then load a few images and create a batch on demand.
# 
# You can view np.memmap as the path collection that can be used to load numpy data on demand when you need to create a batch.
# 
# So let's see how you can work with larger than RAM arrays on disk.

# On my laptop I have only 8GB of RAM.

# I will try to demonstrate how you can handle a 10 GB numpy array dataset in an efficient way. 

# ## Create and save a larger-than-memory array 🥴

# I will now create a large numpy array that doesn't fit in memory. 
# Since I don't have enough RAM, I'll create an empty array on disk, and then load data in chunks that fit in memory.
# 
# ⚠️ If you want to to experiment with large datasets, you may uncomment and run this code. **It will create a ~10GB file on your disk** (we'll delete it at the end of this notebook).
# 
# In my laptop it took me around **2 mins to create the data.**

# In[ ]:


# path = Path('./data')
# X = create_empty_array((100_000, 50, 512), fname='X_on_disk', path=path, mode='r+')

# chunksize = 10_000
# pbar = progress_bar(range(math.ceil(len(X) / chunksize)))
# start = 0
# for i in pbar:
#     end = start + chunksize
#     X[start:end] = np.random.rand(chunksize, X.shape[-2], X.shape[-1])
#     start = end

# # I will create a smaller array. Sinc this fits in memory, I don't need to use a memmap
# y_fn = path/'y_on_disk.npy'
# y = np.random.randint(0, 10, X.shape[0])
# labels = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
# np.save(y_fn, labels[y])

# del X, y


# Ok. So let's check the size of these files if they were in memory.

# In[ ]:


print(f'X array: {os.path.getsize("./data/X_on_disk.npy"):12} bytes ({bytes2str(os.path.getsize("./data/X_on_disk.npy"))})')
print(f'y array: {os.path.getsize("./data/y_on_disk.npy"):12} bytes ({bytes2str(os.path.getsize("./data/y_on_disk.npy"))})')


# ## Load an array on disk (np.memmap) 🧠

# Remember I only have an 8 GB RAM on this laptop, so I couldn't load these datasets in memory.
# 
# ☣️ Actually I accidentally loaded the "X_on_disk.npy" file, and my laptop crashed so I had to reboot it!
# 
# So let's now load data as arrays on disk (np.memmap). The way to do it is super simple, and very efficient. You just do it as you would with a normal array, but add an mmap_mode.
# 
# There are 4 modes: 
# 
# - ‘r’	Open existing file for reading only.
# - ‘r+’	Open existing file for reading and writing.
# - ‘w+’	Create or overwrite existing file for reading and writing.
# - ‘c’	Copy-on-write: assignments affect data in memory, but changes are not saved to disk. The file on disk is read-only.
# 
# I normally use mode 'c' since I want to be able to make changes to data in memory (transforms for example), without affecting data on disk (same approach as with image data). This is the same thing you do with image files on disk, that are just read, and then modified in memory, without changing the file on disk.
# 
# But if you also want to be able to modify data on disk, you can load the array with mmap_mode='r+'.

# In[ ]:


X_on_disk = np.load('./data/X_on_disk.npy', mmap_mode='c')
y_on_disk = np.load('./data/y_on_disk.npy', mmap_mode='c')


# **Fast load**: it only takes a few ms to "load" a memory map to a 10 GB array on disk.
# 
# In fact, the only thing that is loaded is a map to the array stored on disk. That's why it's so fast.

# ## Arrays on disk: main features 📀

# ### Very limited RAM usage

# In[ ]:


print(X_on_disk.shape, y_on_disk.shape)


# In[ ]:


print(f'X array on disk: {sys.getsizeof(X_on_disk):12} bytes ({bytes2str(sys.getsizeof(X_on_disk))})')
print(f'y array on disk: {sys.getsizeof(y_on_disk):12} bytes ({bytes2str(sys.getsizeof(y_on_disk))})')


# **152 bytes of RAM for a 10GB array**. This is the great benefit of arrays on disk.
# 
# Arrays on disk barely use any RAM until they are accessed or sliced and an element is converted into a np.array or a tensor.
# 
# This is equivalent to the size of file paths in images (very limited) compared to the files themselves (actual images). 

# ### Types

# np.memmap is a subclass of np.ndarray

# In[ ]:


isinstance(X_on_disk, np.ndarray)


# In[ ]:


type(X_on_disk)


# ### Operations

# With np.memmap you can perform the same operations you would with a normal numpy array. 
# The most common operations you will perform in deep learning are:
# 
# - slicing
# - calculating stats: mean and std
# - scaling (using normalize or standardize)
# - transformation into a tensor
# 
# Once you get the slice from the array on disk, you'll convert it into a tensor, move to a GPU and performs operations there.

# 
# ⚠️ You need to be careful though not to convert the entire np.memmap to an array/ tensor if it's larger than your RAM. This will crash your computer unless you have enough RAM, so you would have to reboot!
# 
# **DON'T DO THIS:  torch.from_numpy(X) or np.array(X)** unless you have enough RAM.
# 
# To avoid issues during test, I created a smaller array on disk (that I can store in memory). When I want to test something I test it with that array first. It's important to always verify that the type output of your operations is np.memmap, which means data is still in memory.

# #### Slicing

# To ensure you don't bring the entire array in memory (which may crash your computer) you can always work with slices of data, which is by the way how fastai works.
# 
# If you use mode 'r' you can grab a sample and make changes to it, but this won't modify data on disk.

# In[ ]:


x = X_on_disk[0]
x


# It's important to note that **when we perform a math operation on a np.memmap (add, subtract, ...) the output is a np.array, and no longer a np.memmap.**
# 
# ⚠️ Remember you don't want to run this type of operations with a memmap larger than your RAM!! That's why I do it with a slice.

# In[ ]:


x = X_on_disk[0] + 1
x


# In[ ]:


x = torch.from_numpy(X_on_disk[0])
x2 = x + 1
x2


# As you can see, this doesn't affect the original np.memmap

# In[ ]:


X_on_disk[0]


# You can slice an array on disk by any axis, and it'll return a memmap. Slicing by any axis is very fast.

# In[ ]:


X_on_disk[0]


# In[ ]:


X_on_disk[:, 0]


# However, bear in mind that if you use multiple indices, the output will be a regular numpy array. This is important as it will use more RAM. 

# In[ ]:


X_on_disk[[0,1]]


# Unless you use a slice with consecutive indices like this:

# In[ ]:


X_on_disk[:2]


# This continues to be a memmap

# There's a trick we can use avoid this making use of the excellent new L class in fastai. It is to **itemify** the np.memmap/s. 

# In[ ]:


def itemify(*x): return L(*x).zip()


# To itemify one or several np.memmap/s is very fast. Let's see how long it takes with a 10 GB array.

# In[ ]:


X_on_disk_as_items = itemify(X_on_disk)


# 5 seconds to return individual records on disk! Bear in mind you only need to perform this once!
# 
# So now, you can select multiple items at the same time, and they will all still be on disk:

# In[ ]:


X_on_disk_as_items[0,1]


# You can also itemify several items at once: X and y for example. When you slice the list, you'll get tuples.

# In[ ]:


Xy_on_disk_as_items = itemify(X_on_disk, y_on_disk)


# In[ ]:


Xy_on_disk_as_items[0, 1]


# Slicing is very fast, even if there are 100.000 samples.

# In[ ]:


# axis 0
get_ipython().run_line_magic('timeit', 'X_on_disk[0]')


# In[ ]:


# axis 1
get_ipython().run_line_magic('timeit', 'X_on_disk[:, 0]')


# In[ ]:


# axis 2
get_ipython().run_line_magic('timeit', 'X_on_disk[..., 0]')


# In[ ]:


# aixs 0,1
get_ipython().run_line_magic('timeit', 'X_on_disk[0, 0]')


# To compare how fast you can slice a np.memmap, let's create a smaller array that I can fit in memory (X_in_memory). This is 10 times smaller (100 MB) than the one on disk.

# In[ ]:


X_in_memory_small = np.random.rand(10000, 50, 512)


# In[ ]:


get_ipython().run_line_magic('timeit', 'X_in_memory_small[0]')


# Let's create the same array on disk. It's super simple:

# In[ ]:


np.save('./data/X_on_disk_small.npy', X_in_memory_small)
X_on_disk_small = np.load('./data/X_on_disk_small.npy', mmap_mode='c')


# In[ ]:


get_ipython().run_line_magic('timeit', 'X_on_disk_small[0]')


# This is approximately 10 times slower than having arrays on disk, although it's still pretty fast.
# 
# However, if we use the itemified version, it's much faster:

# In[ ]:


get_ipython().run_line_magic('timeit', 'X_on_disk_as_items[0]')


# This is much better! So now you can access 1 of multiple items on disk with a pretty good performance.

# #### Calculating stats: mean and std

# Another benefit of using arrays on disk is that you can calculate the mean and std deviation of the entire dataset. 
# 
# It takes a considerable time since the array is very big (10GB), but it's feasible:
# 
# - mean (0.4999966):  1 min 45 s
# - std  (0.2886839): 11 min 43 s 
# 
# in my laptop. 
# If you need them, you could calculate these stats once, and store the results (similar to ImageNet stats).
# However, you usually need to claculate these metrics for labeled (train) datasets, that tend to be smaller.

# In[ ]:


# X_mean = X_on_disk.mean()
# X_mean


# In[ ]:


# X_std = X_on_disk.std()
# X_std


# #### Conversion into a tensor

# Conversion from an array on disk slice into a tensor is also very fast:

# In[ ]:


torch.from_numpy(X_on_disk[0])


# In[ ]:


X_on_disk_small_0 = X_on_disk_small[0]
X_in_memory_small_0 = X_in_memory_small[0]


# In[ ]:


get_ipython().run_line_magic('timeit', 'torch.from_numpy(X_on_disk_small_0)')


# In[ ]:


get_ipython().run_line_magic('timeit', 'torch.from_numpy(X_in_memory_small_0 )')


# So it takes the same time to convert from numpy.memmap or from a np.array in memory.

# #### Combined operations: slicing plus conversion to tensor

# Let's now check performance of the combined process: slicing plus conversion to a tensor. Based on what we've seen there are 3 options: 
# 
# - slice np.array in memory + conversion to tensor
# - slice np.memmap on disk + conversion to tensor
# - slice itemified np.memmap + converion to tensor

# In[ ]:


get_ipython().run_line_magic('timeit', 'torch.from_numpy(X_in_memory_small[0])')


# In[ ]:


get_ipython().run_line_magic('timeit', 'torch.from_numpy(X_on_disk_small[0])')


# In[ ]:


X_on_disk_small_as_items = itemify(X_on_disk_small)


# In[ ]:


get_ipython().run_line_magic('timeit', 'torch.from_numpy(X_on_disk_small_as_items[0][0])')


# So this last method is **almost as fast as having the array in memory**!! This is an excellent outcome, since slicing arrays in memory is a highly optimized operation. 
# 
# And we have the benefit of having access to very large datasets if needed.

# ## Remove the arrays on disk

# Don't forget to remove the arrays you have created on disk.

# In[ ]:


os.remove('./data/X_on_disk.npy')
os.remove('./data/X_on_disk_small.npy')
os.remove('./data/y_on_disk.npy')


# ## Summary ✅

# We now have a very efficient way to work with very large numpy arrays.
# 
# The process is very simple:
# 
# - create and save the array on disk (as described before)
# - load it with a mmap_mode='c' if you want to be able to modify data in memory but not on dis, or 'r+ if you want to modify data both in memory and on disk.
# 
# So my recommendation would be:
# 
# - use numpy arrays in memory when possible (if your data fits in memory)
# - use numpy memmap (arrays on disk) when data doesn't fit. You will still have a great performance.
