## On this page

  * TSUnwindowedDatasets
  * TSUnwindowedDataset



  * __Report an issue



  1. Data
  2. Unwindowed datasets



# Unwindowed datasets

> Functionality that will allow you to create a dataset that applies sliding windows to the input data on the fly. This heavily reduces the size of the input data files, as only the original unwindowed data needs to be stored.

I’d like to thank both **Thomas Capelle** (https://github.com/tcapelle) and **Xander Dunn** (https://github.com/xanderdunn) for their contributions to make this code possible.

* * *

source

### TSUnwindowedDatasets

> 
>      TSUnwindowedDatasets (dataset, splits)

_Base class for lists with subsets_

* * *

source

### TSUnwindowedDataset

> 
>      TSUnwindowedDataset (X=None, y=None, y_func=None, window_size=1,
>                           stride=1, drop_start=0, drop_end=0, seq_first=True,
>                           **kwargs)

_Initialize self. See help(type(self)) for accurate signature._
    
    
    def y_func(y): return y.astype('float').mean(1)__

This approach works with both univariate and multivariate data.

  * Univariate: we’ll use a simple array with 20 values, one with the seq_len first (X0), the other with seq_len second (X1).
  * Multivariate: we’ll use 2 time series arrays, one with the seq_len first (X2), the other with seq_len second (X3). No sliding window has been applied to them yet.


    
    
    # Univariate
    X0 = np.arange(20).astype(float)
    X1 = np.arange(20).reshape(1, -1).astype(float)
    X0.shape, X0, X1.shape, X1 __
    
    
    ((20,),
     array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
            13., 14., 15., 16., 17., 18., 19.]),
     (1, 20),
     array([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
             13., 14., 15., 16., 17., 18., 19.]]))
    
    
    # Multivariate
    X2 = np.arange(20).reshape(-1,1)*np.array([1, 10, 100]).reshape(1,-1).astype(float)
    X3 = np.arange(20).reshape(1,-1)*np.array([1, 10, 100]).reshape(-1,1).astype(float)
    X2.shape, X3.shape, X2, X3 __
    
    
    ((20, 3),
     (3, 20),
     array([[0.0e+00, 0.0e+00, 0.0e+00],
            [1.0e+00, 1.0e+01, 1.0e+02],
            [2.0e+00, 2.0e+01, 2.0e+02],
            [3.0e+00, 3.0e+01, 3.0e+02],
            [4.0e+00, 4.0e+01, 4.0e+02],
            [5.0e+00, 5.0e+01, 5.0e+02],
            [6.0e+00, 6.0e+01, 6.0e+02],
            [7.0e+00, 7.0e+01, 7.0e+02],
            [8.0e+00, 8.0e+01, 8.0e+02],
            [9.0e+00, 9.0e+01, 9.0e+02],
            [1.0e+01, 1.0e+02, 1.0e+03],
            [1.1e+01, 1.1e+02, 1.1e+03],
            [1.2e+01, 1.2e+02, 1.2e+03],
            [1.3e+01, 1.3e+02, 1.3e+03],
            [1.4e+01, 1.4e+02, 1.4e+03],
            [1.5e+01, 1.5e+02, 1.5e+03],
            [1.6e+01, 1.6e+02, 1.6e+03],
            [1.7e+01, 1.7e+02, 1.7e+03],
            [1.8e+01, 1.8e+02, 1.8e+03],
            [1.9e+01, 1.9e+02, 1.9e+03]]),
     array([[0.0e+00, 1.0e+00, 2.0e+00, 3.0e+00, 4.0e+00, 5.0e+00, 6.0e+00,
             7.0e+00, 8.0e+00, 9.0e+00, 1.0e+01, 1.1e+01, 1.2e+01, 1.3e+01,
             1.4e+01, 1.5e+01, 1.6e+01, 1.7e+01, 1.8e+01, 1.9e+01],
            [0.0e+00, 1.0e+01, 2.0e+01, 3.0e+01, 4.0e+01, 5.0e+01, 6.0e+01,
             7.0e+01, 8.0e+01, 9.0e+01, 1.0e+02, 1.1e+02, 1.2e+02, 1.3e+02,
             1.4e+02, 1.5e+02, 1.6e+02, 1.7e+02, 1.8e+02, 1.9e+02],
            [0.0e+00, 1.0e+02, 2.0e+02, 3.0e+02, 4.0e+02, 5.0e+02, 6.0e+02,
             7.0e+02, 8.0e+02, 9.0e+02, 1.0e+03, 1.1e+03, 1.2e+03, 1.3e+03,
             1.4e+03, 1.5e+03, 1.6e+03, 1.7e+03, 1.8e+03, 1.9e+03]]))

Now, instead of applying SlidingWindow to create and save the time series that can be consumed by a time series model, we can use a dataset that creates the data on the fly. In this way we avoid the need to create and save large files. This approach is also useful when you want to test different sliding window sizes, as otherwise you would need to create files for every size you want to test.The dataset will create the samples correctly formatted and ready to be passed on to a time series architecture.
    
    
    wds0 = TSUnwindowedDataset(X0, window_size=5, stride=2, seq_first=True)[:][0]
    wds1 = TSUnwindowedDataset(X1, window_size=5, stride=2, seq_first=False)[:][0]
    test_eq(wds0, wds1)
    wds0, wds0.data, wds1, wds1.data __
    
    
    (TSTensor(samples:8, vars:1, len:5, device=cpu),
     tensor([[[ 0.,  1.,  2.,  3.,  4.]],
     
             [[ 2.,  3.,  4.,  5.,  6.]],
     
             [[ 4.,  5.,  6.,  7.,  8.]],
     
             [[ 6.,  7.,  8.,  9., 10.]],
     
             [[ 8.,  9., 10., 11., 12.]],
     
             [[10., 11., 12., 13., 14.]],
     
             [[12., 13., 14., 15., 16.]],
     
             [[14., 15., 16., 17., 18.]]]),
     TSTensor(samples:8, vars:1, len:5, device=cpu),
     tensor([[[ 0.,  1.,  2.,  3.,  4.]],
     
             [[ 2.,  3.,  4.,  5.,  6.]],
     
             [[ 4.,  5.,  6.,  7.,  8.]],
     
             [[ 6.,  7.,  8.,  9., 10.]],
     
             [[ 8.,  9., 10., 11., 12.]],
     
             [[10., 11., 12., 13., 14.]],
     
             [[12., 13., 14., 15., 16.]],
     
             [[14., 15., 16., 17., 18.]]]))
    
    
    wds2 = TSUnwindowedDataset(X2, window_size=5, stride=2, seq_first=True)[:][0]
    wds3 = TSUnwindowedDataset(X3, window_size=5, stride=2, seq_first=False)[:][0]
    test_eq(wds2, wds3)
    wds2, wds3, wds2.data, wds3.data __
    
    
    (TSTensor(samples:8, vars:3, len:5, device=cpu),
     TSTensor(samples:8, vars:3, len:5, device=cpu),
     tensor([[[0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00],
              [0.0000e+00, 1.0000e+01, 2.0000e+01, 3.0000e+01, 4.0000e+01],
              [0.0000e+00, 1.0000e+02, 2.0000e+02, 3.0000e+02, 4.0000e+02]],
     
             [[2.0000e+00, 3.0000e+00, 4.0000e+00, 5.0000e+00, 6.0000e+00],
              [2.0000e+01, 3.0000e+01, 4.0000e+01, 5.0000e+01, 6.0000e+01],
              [2.0000e+02, 3.0000e+02, 4.0000e+02, 5.0000e+02, 6.0000e+02]],
     
             [[4.0000e+00, 5.0000e+00, 6.0000e+00, 7.0000e+00, 8.0000e+00],
              [4.0000e+01, 5.0000e+01, 6.0000e+01, 7.0000e+01, 8.0000e+01],
              [4.0000e+02, 5.0000e+02, 6.0000e+02, 7.0000e+02, 8.0000e+02]],
     
             [[6.0000e+00, 7.0000e+00, 8.0000e+00, 9.0000e+00, 1.0000e+01],
              [6.0000e+01, 7.0000e+01, 8.0000e+01, 9.0000e+01, 1.0000e+02],
              [6.0000e+02, 7.0000e+02, 8.0000e+02, 9.0000e+02, 1.0000e+03]],
     
             [[8.0000e+00, 9.0000e+00, 1.0000e+01, 1.1000e+01, 1.2000e+01],
              [8.0000e+01, 9.0000e+01, 1.0000e+02, 1.1000e+02, 1.2000e+02],
              [8.0000e+02, 9.0000e+02, 1.0000e+03, 1.1000e+03, 1.2000e+03]],
     
             [[1.0000e+01, 1.1000e+01, 1.2000e+01, 1.3000e+01, 1.4000e+01],
              [1.0000e+02, 1.1000e+02, 1.2000e+02, 1.3000e+02, 1.4000e+02],
              [1.0000e+03, 1.1000e+03, 1.2000e+03, 1.3000e+03, 1.4000e+03]],
     
             [[1.2000e+01, 1.3000e+01, 1.4000e+01, 1.5000e+01, 1.6000e+01],
              [1.2000e+02, 1.3000e+02, 1.4000e+02, 1.5000e+02, 1.6000e+02],
              [1.2000e+03, 1.3000e+03, 1.4000e+03, 1.5000e+03, 1.6000e+03]],
     
             [[1.4000e+01, 1.5000e+01, 1.6000e+01, 1.7000e+01, 1.8000e+01],
              [1.4000e+02, 1.5000e+02, 1.6000e+02, 1.7000e+02, 1.8000e+02],
              [1.4000e+03, 1.5000e+03, 1.6000e+03, 1.7000e+03, 1.8000e+03]]]),
     tensor([[[0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00],
              [0.0000e+00, 1.0000e+01, 2.0000e+01, 3.0000e+01, 4.0000e+01],
              [0.0000e+00, 1.0000e+02, 2.0000e+02, 3.0000e+02, 4.0000e+02]],
     
             [[2.0000e+00, 3.0000e+00, 4.0000e+00, 5.0000e+00, 6.0000e+00],
              [2.0000e+01, 3.0000e+01, 4.0000e+01, 5.0000e+01, 6.0000e+01],
              [2.0000e+02, 3.0000e+02, 4.0000e+02, 5.0000e+02, 6.0000e+02]],
     
             [[4.0000e+00, 5.0000e+00, 6.0000e+00, 7.0000e+00, 8.0000e+00],
              [4.0000e+01, 5.0000e+01, 6.0000e+01, 7.0000e+01, 8.0000e+01],
              [4.0000e+02, 5.0000e+02, 6.0000e+02, 7.0000e+02, 8.0000e+02]],
     
             [[6.0000e+00, 7.0000e+00, 8.0000e+00, 9.0000e+00, 1.0000e+01],
              [6.0000e+01, 7.0000e+01, 8.0000e+01, 9.0000e+01, 1.0000e+02],
              [6.0000e+02, 7.0000e+02, 8.0000e+02, 9.0000e+02, 1.0000e+03]],
     
             [[8.0000e+00, 9.0000e+00, 1.0000e+01, 1.1000e+01, 1.2000e+01],
              [8.0000e+01, 9.0000e+01, 1.0000e+02, 1.1000e+02, 1.2000e+02],
              [8.0000e+02, 9.0000e+02, 1.0000e+03, 1.1000e+03, 1.2000e+03]],
     
             [[1.0000e+01, 1.1000e+01, 1.2000e+01, 1.3000e+01, 1.4000e+01],
              [1.0000e+02, 1.1000e+02, 1.2000e+02, 1.3000e+02, 1.4000e+02],
              [1.0000e+03, 1.1000e+03, 1.2000e+03, 1.3000e+03, 1.4000e+03]],
     
             [[1.2000e+01, 1.3000e+01, 1.4000e+01, 1.5000e+01, 1.6000e+01],
              [1.2000e+02, 1.3000e+02, 1.4000e+02, 1.5000e+02, 1.6000e+02],
              [1.2000e+03, 1.3000e+03, 1.4000e+03, 1.5000e+03, 1.6000e+03]],
     
             [[1.4000e+01, 1.5000e+01, 1.6000e+01, 1.7000e+01, 1.8000e+01],
              [1.4000e+02, 1.5000e+02, 1.6000e+02, 1.7000e+02, 1.8000e+02],
              [1.4000e+03, 1.5000e+03, 1.6000e+03, 1.7000e+03, 1.8000e+03]]]))

  * __Report an issue


