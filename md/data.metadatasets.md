## On this page

  * TSMetaDatasets
  * TSMetaDataset



  * __Report an issue



  1. Data
  2. Metadataset



# Metadataset

> A dataset of datasets

This functionality will allow you to create a dataset from data stores in multiple, smaller datasets.

I’d like to thank both **Thomas Capelle** (https://github.com/tcapelle) and **Xander Dunn** (https://github.com/xanderdunn) for their contributions to make this code possible.

This functionality allows you to use multiple numpy arrays instead of a single one, which may be very useful in many practical settings. It’s been tested it with 10k+ datasets and it works well.

* * *

source

### TSMetaDatasets

> 
>      TSMetaDatasets (metadataset, splits)

_Base class for lists with subsets_

* * *

source

### TSMetaDataset

> 
>      TSMetaDataset (dataset_list, **kwargs)

_Initialize self. See help(type(self)) for accurate signature._

Let’s create 3 datasets. In this case they will have different sizes.
    
    
    vocab = alphabet[:10]
    dsets = []
    for i in range(3):
        size = np.random.randint(50, 150)
        X = torch.rand(size, 5, 50)
        y = vocab[torch.randint(0, 10, (size,))]
        tfms = [None, TSClassification(vocab=vocab)]
        dset = TSDatasets(X, y, tfms=tfms)
        dsets.append(dset)
    
    
    
    metadataset = TSMetaDataset(dsets)
    splits = TimeSplitter(show_plot=False)(metadataset)
    metadatasets = TSMetaDatasets(metadataset, splits=splits)
    dls = TSDataLoaders.from_dsets(metadatasets.train, metadatasets.valid)
    xb, yb = dls.train.one_batch()
    xb, yb __
    
    
    (TSTensor(samples:64, vars:5, len:50, device=cpu, dtype=torch.float32),
     TensorCategory([1, 0, 3, 9, 7, 2, 8, 6, 1, 1, 1, 8, 1, 1, 9, 2, 6, 6, 1, 5, 5,
                     6, 9, 2, 7, 1, 6, 4, 9, 2, 5, 0, 4, 9, 1, 4, 4, 6, 0, 8, 8, 5,
                     8, 6, 9, 0, 8, 8, 6, 4, 8, 9, 7, 3, 4, 7, 7, 8, 6, 2, 3, 0, 7,
                     4]))

You can train metadatasets as you would train any other time series model in `tsai`:
    
    
    learn = ts_learner(dls, arch="TSTPlus")
    learn.fit_one_cycle(1)
    learn.export("test.pkl")__

For inference, you should create the new metadatasets using the same method you used when you trained it. The you use fastai’s learn.get_preds method to generate predictions:
    
    
    vocab = alphabet[:10]
    dsets = []
    for i in range(3):
        size = np.random.randint(50, 150)
        X = torch.rand(size, 5, 50)
        y = vocab[torch.randint(0, 10, (size,))]
        tfms = [None, TSClassification(vocab=vocab)]
        dset = TSDatasets(X, y, tfms=tfms)
        dsets.append(dset)
    metadataset = TSMetaDataset(dsets)
    dl = TSDataLoader(metadataset)
    
    
    learn = load_learner("test.pkl")
    learn.get_preds(dl=dl)__

There also en easy way to map any particular sample in a batch to the original dataset and id:
    
    
    dls = TSDataLoaders.from_dsets(metadatasets.train, metadatasets.valid)
    xb, yb = first(dls.train)
    mappings = dls.train.dataset.mapping_idxs
    for i, (xbi, ybi) in enumerate(zip(xb, yb)):
        ds, idx = mappings[i]
        test_close(dsets[ds][idx][0].data.cpu(), xbi.cpu())
        test_close(dsets[ds][idx][1].data.cpu(), ybi.cpu())__

For example the 3rd sample in this batch would be:
    
    
    dls.train.dataset.mapping_idxs[2]__
    
    
    array([  0, 112], dtype=int32)

  * __Report an issue


