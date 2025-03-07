## On this page

  * get_tabular_ds
  * get_tabular_dls
  * preprocess_df



  * __Report an issue



  1. Data
  2. Time Series Tabular Data



# Time Series Tabular Data

> Main Tabular functions used throughout the library. This is helpful when you have additional time series data like metadata, time series features, etc.

* * *

source

### get_tabular_ds

> 
>      get_tabular_ds (df, procs=[<class 'fastai.tabular.core.Categorify'>,
>                      <class 'fastai.tabular.core.FillMissing'>, <class
>                      'fastai.data.transforms.Normalize'>], cat_names=None,
>                      cont_names=None, y_names=None, groupby=None,
>                      y_block=None, splits=None, do_setup=True, inplace=False,
>                      reduce_memory=True, device=None)

* * *

source

### get_tabular_dls

> 
>      get_tabular_dls (df, procs=[<class 'fastai.tabular.core.Categorify'>,
>                       <class 'fastai.tabular.core.FillMissing'>, <class
>                       'fastai.data.transforms.Normalize'>], cat_names=None,
>                       cont_names=None, y_names=None, bs=64, y_block=None,
>                       splits=None, do_setup=True, inplace=False,
>                       reduce_memory=True, device=None, path:str|Path='.')

| **Type** | **Default** | **Details**  
---|---|---|---  
df |  |  |   
procs | list | [<class ‘fastai.tabular.core.Categorify’>, <class ‘fastai.tabular.core.FillMissing’>, <class ‘fastai.data.transforms.Normalize’>] |   
cat_names | NoneType | None |   
cont_names | NoneType | None |   
y_names | NoneType | None |   
bs | int | 64 |   
y_block | NoneType | None |   
splits | NoneType | None |   
do_setup | bool | True |   
inplace | bool | False |   
reduce_memory | bool | True |   
device | NoneType | None | Device to put `DataLoaders`  
path | str | pathlib.Path | . | Path to store export objects  
  
* * *

source

### preprocess_df

> 
>      preprocess_df (df, procs=[<class 'fastai.tabular.core.Categorify'>,
>                     <class 'fastai.tabular.core.FillMissing'>, <class
>                     'fastai.data.transforms.Normalize'>], cat_names=None,
>                     cont_names=None, y_names=None, sample_col=None,
>                     reduce_memory=True)
    
    
    path = untar_data(URLs.ADULT_SAMPLE)
    df = pd.read_csv(path/'adult.csv')
    # df['salary'] = np.random.rand(len(df)) # uncomment to simulate a cont dependent variable
    
    cat_names = ['workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                 'capital-gain', 'capital-loss', 'native-country']
    cont_names = ['age', 'fnlwgt', 'hours-per-week']
    target = ['salary']
    splits = RandomSplitter()(range_of(df))
    
    dls = get_tabular_dls(df, cat_names=cat_names, cont_names=cont_names, y_names='salary', splits=splits, bs=512, device=device)
    dls.show_batch()__

| workclass | education | education-num | marital-status | occupation | relationship | race | sex | capital-gain | capital-loss | native-country | age | fnlwgt | hours-per-week | salary  
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---  
0 | Private | Some-college | 10.0 | Divorced | Exec-managerial | Not-in-family | White | Male | 0 | 0 | United-States | 48.000000 | 190072.000005 | 50.000000 | >=50k  
1 | Self-emp-not-inc | Some-college | 10.0 | Married-civ-spouse | Sales | Husband | White | Male | 0 | 0 | United-States | 72.000001 | 284120.002964 | 40.000000 | <50k  
2 | Private | Some-college | 10.0 | Married-civ-spouse | Protective-serv | Husband | Black | Male | 0 | 0 | United-States | 72.000001 | 53684.002497 | 40.000000 | <50k  
3 | Self-emp-inc | Some-college | 10.0 | Married-civ-spouse | Farming-fishing | Husband | White | Male | 0 | 0 | United-States | 47.000000 | 337049.998875 | 40.000000 | <50k  
4 | Private | HS-grad | 9.0 | Divorced | Craft-repair | Not-in-family | White | Male | 0 | 0 | United-States | 46.000000 | 207677.000707 | 30.000000 | <50k  
5 | Private | 5th-6th | 3.0 | Divorced | Priv-house-serv | Unmarried | White | Female | 0 | 0 | Mexico | 45.000000 | 265082.999142 | 35.000000 | <50k  
6 | Private | Assoc-acdm | 12.0 | Never-married | Other-service | Not-in-family | White | Female | 0 | 0 | United-States | 28.000000 | 150296.001328 | 79.999999 | <50k  
7 | Private | HS-grad | 9.0 | Married-civ-spouse | Exec-managerial | Husband | White | Male | 0 | 0 | United-States | 50.000000 | 94080.999353 | 40.000000 | >=50k  
8 | Private | Assoc-voc | 11.0 | Married-civ-spouse | Exec-managerial | Husband | White | Male | 0 | 0 | Germany | 58.000000 | 235624.000302 | 40.000000 | >=50k  
9 | Private | HS-grad | 9.0 | Never-married | Other-service | Unmarried | Black | Female | 0 | 0 | Japan | 29.000000 | 419721.008996 | 40.000000 | <50k  
      
    
    metrics = mae if dls.c == 1 else accuracy
    learn = tabular_learner(dls, layers=[200, 100], y_range=None, metrics=metrics)
    learn.fit(1, 1e-2)__

epoch | train_loss | valid_loss | accuracy | time  
---|---|---|---|---  
0 | 0.349525 | 0.288922 | 0.866093 | 00:05  
      
    
    learn.dls.one_batch()__
    
    
    (tensor([[  5,  12,   9,  ...,   1,   1,  21],
             [  1,  10,  13,  ...,   1,   1,   3],
             [  5,   4,   2,  ...,   1,   1,   6],
             ...,
             [  5,   6,   4,  ...,   1,   1,  40],
             [  3,  10,  13,  ...,   1,   1,  40],
             [  5,  12,   9,  ..., 116,   1,  40]]),
     tensor([[-0.2593,  0.1234,  1.1829],
             [-0.9913, -1.4041, -0.0347],
             [-0.1129,  0.4583, -0.0347],
             ...,
             [-1.5769, -0.1989,  0.3712],
             [ 0.4727, -1.4400,  0.3712],
             [ 1.5708, -0.2222, -0.0347]]),
     tensor([[1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [1],
             [1],
             [1],
             [0],
             [0],
             [1],
             [1],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [1],
             [1],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [1],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [1],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [1],
             [0],
             [1],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [1],
             [0],
             [1],
             [0],
             [0],
             [1],
             [0],
             [1],
             [1],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [1],
             [1],
             [1],
             [1],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [1],
             [1],
             [0],
             [0],
             [0],
             [1],
             [1],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [1],
             [0],
             [1],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [1],
             [1],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [1],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [1],
             [0],
             [1],
             [0],
             [1],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [1],
             [1],
             [1],
             [0],
             [0],
             [0],
             [1],
             [1],
             [1],
             [0],
             [1],
             [1],
             [0],
             [1],
             [1],
             [1],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [1],
             [0],
             [1],
             [1],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [1],
             [0],
             [1],
             [1],
             [1],
             [0],
             [1],
             [0],
             [1],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [1],
             [1],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [1],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [1],
             [1],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [1],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [0],
             [0],
             [1],
             [0],
             [0],
             [1],
             [1]], dtype=torch.int8))
    
    
    learn.model __
    
    
    TabularModel(
      (embeds): ModuleList(
        (0): Embedding(10, 6)
        (1): Embedding(17, 8)
        (2): Embedding(17, 8)
        (3): Embedding(8, 5)
        (4): Embedding(16, 8)
        (5): Embedding(7, 5)
        (6): Embedding(6, 4)
        (7): Embedding(3, 3)
        (8): Embedding(117, 23)
        (9): Embedding(90, 20)
        (10): Embedding(43, 13)
      )
      (emb_drop): Dropout(p=0.0, inplace=False)
      (bn_cont): BatchNorm1d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (layers): Sequential(
        (0): LinBnDrop(
          (0): Linear(in_features=106, out_features=200, bias=False)
          (1): ReLU(inplace=True)
          (2): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): LinBnDrop(
          (0): Linear(in_features=200, out_features=100, bias=False)
          (1): ReLU(inplace=True)
          (2): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): LinBnDrop(
          (0): Linear(in_features=100, out_features=2, bias=True)
        )
      )
    )
    
    
    path = untar_data(URLs.ADULT_SAMPLE)
    df = pd.read_csv(path/'adult.csv')
    cat_names = ['workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                 'capital-gain', 'capital-loss', 'native-country']
    cont_names = ['age', 'fnlwgt', 'hours-per-week']
    target = ['salary']
    df, procs = preprocess_df(df, procs=[Categorify, FillMissing, Normalize], cat_names=cat_names, cont_names=cont_names, y_names=target, 
                              sample_col=None, reduce_memory=True)
    df.head()__

| workclass | education | education-num | marital-status | occupation | relationship | race | sex | capital-gain | capital-loss | native-country | age | fnlwgt | hours-per-week | salary  
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---  
0 | 5 | 8 | 12 | 3 | 0 | 6 | 5 | 1 | 1 | 48 | 40 | 0.763796 | -0.838084 | -0.035429 | 1  
1 | 5 | 13 | 14 | 1 | 5 | 2 | 5 | 2 | 101 | 1 | 40 | 0.397233 | 0.444987 | 0.369519 | 1  
2 | 5 | 12 | 0 | 1 | 0 | 5 | 3 | 1 | 1 | 1 | 40 | -0.042642 | -0.886734 | -0.683348 | 0  
3 | 6 | 15 | 15 | 3 | 11 | 1 | 2 | 2 | 1 | 1 | 40 | -0.042642 | -0.728873 | -0.035429 | 1  
4 | 7 | 6 | 0 | 3 | 9 | 6 | 3 | 1 | 1 | 1 | 40 | 0.250608 | -1.018314 | 0.774468 | 0  
      
    
    procs.classes, procs.means, procs.stds __
    
    
    ({'workclass': ['#na#', ' ?', ' Federal-gov', ' Local-gov', ' Never-worked', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay'],
      'education': ['#na#', ' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college'],
      'education-num': ['#na#', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
      'marital-status': ['#na#', ' Divorced', ' Married-AF-spouse', ' Married-civ-spouse', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed'],
      'occupation': ['#na#', ' ?', ' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving'],
      'relationship': ['#na#', ' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife'],
      'race': ['#na#', ' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White'],
      'sex': ['#na#', ' Female', ' Male'],
      'capital-gain': ['#na#', 0, 114, 401, 594, 914, 991, 1055, 1086, 1111, 1151, 1173, 1409, 1424, 1455, 1471, 1506, 1639, 1797, 1831, 1848, 2009, 2036, 2050, 2062, 2105, 2174, 2176, 2202, 2228, 2290, 2329, 2346, 2354, 2387, 2407, 2414, 2463, 2538, 2580, 2597, 2635, 2653, 2829, 2885, 2907, 2936, 2961, 2964, 2977, 2993, 3103, 3137, 3273, 3325, 3411, 3418, 3432, 3456, 3464, 3471, 3674, 3781, 3818, 3887, 3908, 3942, 4064, 4101, 4386, 4416, 4508, 4650, 4687, 4787, 4865, 4931, 4934, 5013, 5060, 5178, 5455, 5556, 5721, 6097, 6360, 6418, 6497, 6514, 6723, 6767, 6849, 7298, 7430, 7443, 7688, 7896, 7978, 8614, 9386, 9562, 10520, 10566, 10605, 11678, 13550, 14084, 14344, 15020, 15024, 15831, 18481, 20051, 22040, 25124, 25236, 27828, 34095, 41310, 99999],
      'capital-loss': ['#na#', 0, 155, 213, 323, 419, 625, 653, 810, 880, 974, 1092, 1138, 1258, 1340, 1380, 1408, 1411, 1485, 1504, 1539, 1564, 1573, 1579, 1590, 1594, 1602, 1617, 1628, 1648, 1651, 1668, 1669, 1672, 1719, 1721, 1726, 1735, 1740, 1741, 1755, 1762, 1816, 1825, 1844, 1848, 1876, 1887, 1902, 1944, 1974, 1977, 1980, 2001, 2002, 2042, 2051, 2057, 2080, 2129, 2149, 2163, 2174, 2179, 2201, 2205, 2206, 2231, 2238, 2246, 2258, 2267, 2282, 2339, 2352, 2377, 2392, 2415, 2444, 2457, 2467, 2472, 2489, 2547, 2559, 2603, 2754, 2824, 3004, 3683, 3770, 3900, 4356],
      'native-country': ['#na#', ' ?', ' Cambodia', ' Canada', ' China', ' Columbia', ' Cuba', ' Dominican-Republic', ' Ecuador', ' El-Salvador', ' England', ' France', ' Germany', ' Greece', ' Guatemala', ' Haiti', ' Holand-Netherlands', ' Honduras', ' Hong', ' Hungary', ' India', ' Iran', ' Ireland', ' Italy', ' Jamaica', ' Japan', ' Laos', ' Mexico', ' Nicaragua', ' Outlying-US(Guam-USVI-etc)', ' Peru', ' Philippines', ' Poland', ' Portugal', ' Puerto-Rico', ' Scotland', ' South', ' Taiwan', ' Thailand', ' Trinadad&Tobago', ' United-States', ' Vietnam', ' Yugoslavia']},
     {'age': 38.58164675532078,
      'fnlwgt': 189778.36651208502,
      'hours-per-week': 40.437455852092995},
     {'age': 13.640223192304274,
      'fnlwgt': 105548.3568809908,
      'hours-per-week': 12.347239175707989})

  * __Report an issue


