## On this page

  * TabHead
  * TabBackbone
  * TabModel



  * __Report an issue



  1. Models
  2. Tabular models
  3. TabModel



# TabModel

This is an implementation created by Ignacio Oguiza (oguiza@timeseriesAI.co) based on fastai’s TabularModel.

We built it so that it’s easy to change the head of the model, something that is particularly interesting when building hybrid models.

* * *

source

### TabHead

> 
>      TabHead (emb_szs, n_cont, c_out, layers=None, fc_dropout=None,
>               y_range=None, use_bn=True, bn_final=False, lin_first=False,
>               act=ReLU(inplace=True), skip=False)

_Basic head for tabular data._

* * *

source

### TabBackbone

> 
>      TabBackbone (emb_szs, n_cont, embed_p=0.0, bn_cont=True)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### TabModel

> 
>      TabModel (emb_szs, n_cont, c_out, layers=None, fc_dropout=None,
>                embed_p=0.0, y_range=None, use_bn=True, bn_final=False,
>                bn_cont=True, lin_first=False, act=ReLU(inplace=True),
>                skip=False)

_Basic model for tabular data._
    
    
    from fastai.tabular.core import *
    from tsai.data.tabular import *__
    
    
    path = untar_data(URLs.ADULT_SAMPLE)
    df = pd.read_csv(path/'adult.csv')
    # df['salary'] = np.random.rand(len(df)) # uncomment to simulate a cont dependent variable
    procs = [Categorify, FillMissing, Normalize]
    cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
    cont_names = ['age', 'fnlwgt', 'education-num']
    y_names = ['salary']
    y_block = RegressionBlock() if isinstance(df['salary'].values[0], float) else CategoryBlock()
    splits = RandomSplitter()(range_of(df))
    pd.options.mode.chained_assignment=None
    to = TabularPandas(df, procs=procs, cat_names=cat_names, cont_names=cont_names, y_names=y_names, y_block=y_block, splits=splits, inplace=True, 
                       reduce_memory=False)
    to.show(5)
    tab_dls = to.dataloaders(bs=16, val_bs=32)
    b = first(tab_dls.train)
    test_eq((b[0].shape, b[1].shape, b[2].shape), (torch.Size([16, 7]), torch.Size([16, 3]), torch.Size([16, 1])))__

| workclass | education | marital-status | occupation | relationship | race | education-num_na | age | fnlwgt | education-num | salary  
---|---|---|---|---|---|---|---|---|---|---|---  
20505 | Private | HS-grad | Married-civ-spouse | Sales | Husband | White | False | 47.0 | 197836.0 | 9.0 | <50k  
28679 | Private | HS-grad | Married-civ-spouse | Craft-repair | Husband | White | False | 28.0 | 65078.0 | 9.0 | >=50k  
11669 | Private | HS-grad | Never-married | Adm-clerical | Not-in-family | White | False | 38.0 | 202683.0 | 9.0 | <50k  
29079 | Self-emp-not-inc | Bachelors | Married-civ-spouse | Prof-specialty | Husband | White | False | 41.0 | 168098.0 | 13.0 | <50k  
7061 | Private | HS-grad | Married-civ-spouse | Adm-clerical | Husband | White | False | 31.0 | 243442.0 | 9.0 | <50k  
      
    
    tab_model = build_tabular_model(TabModel, dls=tab_dls)
    b = first(tab_dls.train)
    test_eq(tab_model.to(b[0].device)(*b[:-1]).shape, (tab_dls.bs, tab_dls.c))
    learn = Learner(tab_dls, tab_model, splitter=ts_splitter)
    p1 = count_parameters(learn.model)
    learn.freeze()
    p2 = count_parameters(learn.model)
    learn.unfreeze()
    p3 = count_parameters(learn.model)
    assert p1 == p3
    assert p1 > p2 > 0 __

  * __Report an issue


