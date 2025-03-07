## On this page

  * MixedDataLoaders
  * MixedDataLoader
  * get_mixed_dls



  * __Report an issue



  1. Data
  2. Mixed data



# Mixed data

> DataLoader than can take data from multiple dataloaders with different types of data

* * *

source

### MixedDataLoaders

> 
>      MixedDataLoaders (*loaders, path:str|pathlib.Path='.', device=None)

_Basic wrapper around several`DataLoader`s._

| **Type** | **Default** | **Details**  
---|---|---|---  
loaders | VAR_POSITIONAL |  | `DataLoader` objects to wrap  
path | str | pathlib.Path | . | Path to store export objects  
device | NoneType | None | Device to put `DataLoaders`  
  
* * *

source

### MixedDataLoader

> 
>      MixedDataLoader (*loaders, path='.', shuffle=False, device=None, bs=None)

_Accepts any number of`DataLoader` and a device_

* * *

source

### get_mixed_dls

> 
>      get_mixed_dls (*dls, device=None, shuffle_train=None, shuffle_valid=None,
>                     **kwargs)
    
    
    from tsai.data.tabular import *__
    
    
    path = untar_data(URLs.ADULT_SAMPLE)
    df = pd.read_csv(path/'adult.csv')
    # df['salary'] = np.random.rand(len(df)) # uncomment to simulate a cont dependent variable
    target = 'salary'
    splits = RandomSplitter()(range_of(df))
    
    cat_names = ['workclass', 'education', 'marital-status']
    cont_names = ['age', 'fnlwgt']
    dls1 = get_tabular_dls(df, cat_names=cat_names, cont_names=cont_names, y_names=target, splits=splits, bs=512)
    dls1.show_batch()
    
    cat_names = None #['occupation', 'relationship', 'race']
    cont_names = ['education-num']
    dls2 = get_tabular_dls(df, cat_names=cat_names, cont_names=cont_names, y_names=target, splits=splits, bs=128)
    dls2.show_batch()__

| workclass | education | marital-status | age | fnlwgt | salary  
---|---|---|---|---|---|---  
0 | Private | 5th-6th | Separated | 47.000000 | 225065.000159 | <50k  
1 | Private | HS-grad | Married-civ-spouse | 56.999999 | 84887.999356 | <50k  
2 | Private | Assoc-voc | Married-civ-spouse | 30.000000 | 176409.999275 | >=50k  
3 | Private | Some-college | Married-civ-spouse | 31.000000 | 232474.999969 | <50k  
4 | Private | 10th | Married-civ-spouse | 26.000000 | 293984.002897 | <50k  
5 | Private | HS-grad | Married-civ-spouse | 54.000000 | 167770.000370 | >=50k  
6 | Private | Bachelors | Never-married | 25.000000 | 60357.998190 | <50k  
7 | Local-gov | 7th-8th | Married-civ-spouse | 62.000000 | 203524.999993 | <50k  
8 | Private | Some-college | Married-civ-spouse | 36.000000 | 220510.999048 | <50k  
9 | State-gov | Doctorate | Married-civ-spouse | 55.000000 | 120781.002923 | >=50k  
  
| education-num_na | education-num | salary  
---|---|---|---  
0 | False | 10.0 | <50k  
1 | False | 9.0 | <50k  
2 | False | 13.0 | <50k  
3 | False | 13.0 | >=50k  
4 | False | 10.0 | <50k  
5 | False | 9.0 | <50k  
6 | False | 9.0 | <50k  
7 | False | 10.0 | <50k  
8 | False | 14.0 | >=50k  
9 | False | 9.0 | <50k  
      
    
    dls = get_mixed_dls(dls1, dls2, bs=8)
    first(dls.train)
    first(dls.valid)
    torch.save(dls,'export/mixed_dls.pth')
    del dls
    dls = torch.load('export/mixed_dls.pth')
    dls.train.show_batch()__

| workclass | education | marital-status | age | fnlwgt | salary  
---|---|---|---|---|---|---  
0 | Private | Masters | Divorced | 44.000000 | 236746.000153 | >=50k  
1 | Private | Bachelors | Never-married | 22.000000 | 189950.000000 | <50k  
2 | Private | HS-grad | Married-civ-spouse | 56.999999 | 120302.001777 | <50k  
3 | Private | HS-grad | Never-married | 29.000000 | 131087.999775 | <50k  
4 | Self-emp-not-inc | HS-grad | Never-married | 35.000000 | 179171.000276 | <50k  
5 | Self-emp-not-inc | HS-grad | Divorced | 75.000001 | 242107.999406 | <50k  
6 | Private | 12th | Never-married | 36.000000 | 137420.999182 | <50k  
7 | Private | Doctorate | Married-civ-spouse | 35.000000 | 189623.000011 | >=50k  
  
| education-num_na | education-num | salary  
---|---|---|---  
0 | False | 9.0 | <50k  
1 | False | 9.0 | <50k  
2 | False | 9.0 | <50k  
3 | False | 9.0 | <50k  
4 | False | 9.0 | <50k  
5 | False | 10.0 | <50k  
6 | False | 9.0 | <50k  
7 | False | 10.0 | >=50k  
      
    
    from tsai.data.validation import TimeSplitter
    from tsai.data.core import TSRegression, get_ts_dls __
    
    
    X = np.repeat(np.repeat(np.arange(16)[:, None, None], 2, 1), 5, 2).astype(float)
    y = np.concatenate([np.arange(len(X)//2)]*2)
    alphabet = np.array(list(string.ascii_lowercase))
    # y = alphabet[y]
    splits = TimeSplitter(.5, show_plot=False)(range_of(X))
    tfms = [None, TSRegression()]
    dls1 = get_ts_dls(X, y, splits=splits, tfms=tfms, bs=4)
    for xb, yb in iter(dls1.train):
        print(xb.data, yb)__
    
    
    tensor([[[5., 5., 5., 5., 5.],
             [5., 5., 5., 5., 5.]],
    
            [[0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]],
    
            [[4., 4., 4., 4., 4.],
             [4., 4., 4., 4., 4.]],
    
            [[3., 3., 3., 3., 3.],
             [3., 3., 3., 3., 3.]]], device='mps:0') tensor([5., 0., 4., 3.], device='mps:0')
    tensor([[[6., 6., 6., 6., 6.],
             [6., 6., 6., 6., 6.]],
    
            [[1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.]],
    
            [[2., 2., 2., 2., 2.],
             [2., 2., 2., 2., 2.]],
    
            [[7., 7., 7., 7., 7.],
             [7., 7., 7., 7., 7.]]], device='mps:0') tensor([6., 1., 2., 7.], device='mps:0')
    
    
    data = np.repeat(np.arange(16)[:, None], 3, 1)*np.array([1, 10, 100])
    df = pd.DataFrame(data, columns=['cat1', 'cat2', 'cont'])
    df['cont'] = df['cont'].astype(float)
    df['target'] = y
    display(df)
    cat_names = ['cat1', 'cat2']
    cont_names = ['cont']
    target = 'target'
    dls2 = get_tabular_dls(df, procs=[Categorify, FillMissing, #Normalize
                                     ], cat_names=cat_names, cont_names=cont_names, y_names=target, splits=splits, bs=4)
    for b in iter(dls2.train):
        print(b[0], b[1], b[2])__

| cat1 | cat2 | cont | target  
---|---|---|---|---  
0 | 0 | 0 | 0.0 | 0  
1 | 1 | 10 | 100.0 | 1  
2 | 2 | 20 | 200.0 | 2  
3 | 3 | 30 | 300.0 | 3  
4 | 4 | 40 | 400.0 | 4  
5 | 5 | 50 | 500.0 | 5  
6 | 6 | 60 | 600.0 | 6  
7 | 7 | 70 | 700.0 | 7  
8 | 8 | 80 | 800.0 | 0  
9 | 9 | 90 | 900.0 | 1  
10 | 10 | 100 | 1000.0 | 2  
11 | 11 | 110 | 1100.0 | 3  
12 | 12 | 120 | 1200.0 | 4  
13 | 13 | 130 | 1300.0 | 5  
14 | 14 | 140 | 1400.0 | 6  
15 | 15 | 150 | 1500.0 | 7  
      
    
    tensor([[5, 5],
            [5, 5],
            [6, 6],
            [4, 4]], device='mps:0') tensor([[400.],
            [400.],
            [500.],
            [300.]], device='mps:0') tensor([[4],
            [4],
            [5],
            [3]], device='mps:0', dtype=torch.int8)
    tensor([[4, 4],
            [7, 7],
            [2, 2],
            [1, 1]], device='mps:0') tensor([[300.],
            [600.],
            [100.],
            [  0.]], device='mps:0') tensor([[3],
            [6],
            [1],
            [0]], device='mps:0', dtype=torch.int8)
    
    
    bs = 8
    dls = get_mixed_dls(dls1, dls2, bs=bs)
    dl = dls.train
    xb, yb = dl.one_batch()
    test_eq(len(xb), 2)
    test_eq(len(xb[0]), bs)
    test_eq(len(xb[1]), 2)
    test_eq(len(xb[1][0]), bs)
    test_eq(len(xb[1][1]), bs)
    test_eq(xb[0].data[:, 0, 0].long(), xb[1][0][:, 0] - 1) # categorical data and ts are in synch
    test_eq(xb[0].data[:, 0, 0], (xb[1][1]/100).flatten()) # continuous data and ts are in synch
    test_eq(tensor(dl.input_idxs), yb.long().cpu())
    dl = dls.valid
    xb, yb = dl.one_batch()
    test_eq(tensor(y[dl.input_idxs]), yb.long().cpu())__
    
    
    bs = 4
    dls = get_mixed_dls(dls1, dls2, bs=bs)
    for xb, yb in iter(dls.train):
        print(xb[0].data, xb[1], yb)__
    
    
    tensor([[[0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]],
    
            [[1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.]],
    
            [[2., 2., 2., 2., 2.],
             [2., 2., 2., 2., 2.]],
    
            [[4., 4., 4., 4., 4.],
             [4., 4., 4., 4., 4.]]], device='mps:0') (tensor([[1, 1],
            [2, 2],
            [3, 3],
            [5, 5]], device='mps:0'), tensor([[  0.],
            [100.],
            [200.],
            [400.]], device='mps:0')) tensor([0., 1., 2., 4.], device='mps:0')
    tensor([[[3., 3., 3., 3., 3.],
             [3., 3., 3., 3., 3.]],
    
            [[5., 5., 5., 5., 5.],
             [5., 5., 5., 5., 5.]],
    
            [[6., 6., 6., 6., 6.],
             [6., 6., 6., 6., 6.]],
    
            [[7., 7., 7., 7., 7.],
             [7., 7., 7., 7., 7.]]], device='mps:0') (tensor([[4, 4],
            [6, 6],
            [7, 7],
            [8, 8]], device='mps:0'), tensor([[300.],
            [500.],
            [600.],
            [700.]], device='mps:0')) tensor([3., 5., 6., 7.], device='mps:0')

  * __Report an issue


