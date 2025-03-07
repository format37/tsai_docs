## On this page

  * Description
  * What’s new:
  * Installation
    * Pip install
    * Conda install
  * Documentation
  * Available models:
  * How to start using tsai?
  * Examples
    * Binary, univariate classification
    * Multi-class, multivariate classification
    * Multivariate Regression
    * Forecasting
      * Single step
      * Multi-step
  * Input data format
  * How to contribute to tsai?
  * Enterprise support and consulting services:
  * Citing tsai



  * __Report an issue



# tsai

  
  


## Description

> State-of-the-art Deep Learning library for Time Series and Sequences.

`tsai` is an open-source deep learning package built on top of Pytorch & fastai focused on state-of-the-art techniques for time series tasks like classification, regression, forecasting, imputation…

`tsai` is currently under active development by timeseriesAI.

## What’s new:

During the last few releases, here are some of the most significant additions to `tsai`:

  * **New models** : PatchTST (Accepted by ICLR 2023), RNN with Attention (RNNAttention, LSTMAttention, GRUAttention), TabFusionTransformer, …
  * **New datasets** : we have increased the number of datasets you can download using `tsai`: 
    * 128 univariate classification datasets
    * 30 multivariate classification datasets
    * 15 regression datasets
    * 62 forecasting datasets
    * 9 long term forecasting datasets
  * **New tutorials** : PatchTST. Based on some of your requests, we are planning to release additional tutorials on data preparation and forecasting.
  * **New functionality** : sklearn-type pipeline transforms, walk-foward cross validation, reduced RAM requirements, and a lot of new functionality to perform more accurate time series forecasts.
  * Pytorch 2.0 support.



## Installation

### Pip install

You can install the **latest stable** version from pip using:
    
    
    pip install tsai __

If you plan to develop tsai yourself, or want to be on the cutting edge, you can use an editable install. First install PyTorch, and then:
    
    
    git clone https://github.com/timeseriesAI/tsai
    pip install -e "tsai[dev]"__

Note: starting with tsai 0.3.0 tsai will only install hard dependencies. Other soft dependencies (which are only required for selected tasks) will not be installed by default (this is the recommended approach. If you require any of the dependencies that is not installed, tsai will ask you to install it when necessary). If you still want to install tsai with all its dependencies you can do it by running:
    
    
    pip install tsai[extras]__

### Conda install

You can also install tsai using conda (note that if you replace conda with mamba the install process will be much faster and more reliable):
    
    
    conda install -c timeseriesai tsai __

## Documentation

Here’s the link to the documentation.

## Available models:

Here’s a list with some of the state-of-the-art models available in `tsai`:

  * LSTM (Hochreiter, 1997) (paper)
  * GRU (Cho, 2014) (paper)
  * MLP \- Multilayer Perceptron (Wang, 2016) (paper)
  * FCN \- Fully Convolutional Network (Wang, 2016) (paper)
  * ResNet \- Residual Network (Wang, 2016) (paper)
  * LSTM-FCN (Karim, 2017) (paper)
  * GRU-FCN (Elsayed, 2018) (paper)
  * mWDN \- Multilevel wavelet decomposition network (Wang, 2018) (paper)
  * TCN \- Temporal Convolutional Network (Bai, 2018) (paper)
  * MLSTM-FCN \- Multivariate LSTM-FCN (Karim, 2019) (paper)
  * InceptionTime (Fawaz, 2019) (paper)
  * Rocket (Dempster, 2019) (paper)
  * XceptionTime (Rahimian, 2019) (paper)
  * ResCNN \- 1D-ResCNN (Zou , 2019) (paper)
  * TabModel \- modified from fastai’s TabularModel
  * OmniScale \- Omni-Scale 1D-CNN (Tang, 2020) (paper)
  * TST \- Time Series Transformer (Zerveas, 2020) (paper)
  * TabTransformer (Huang, 2020) (paper)
  * TSiT Adapted from ViT (Dosovitskiy, 2020) (paper)
  * MiniRocket (Dempster, 2021) (paper)
  * XCM \- An Explainable Convolutional Neural Network (Fauvel, 2021) (paper)
  * gMLP \- Gated Multilayer Perceptron (Liu, 2021) (paper)
  * TSPerceiver \- Adapted from Perceiver IO (Jaegle, 2021) (paper)
  * GatedTabTransformer (Cholakov, 2022) (paper)
  * TSSequencerPlus \- Adapted from Sequencer (Tatsunami, 2022) (paper)
  * PatchTST \- (Nie, 2022) (paper)



plus other custom models like: TransformerModel, LSTMAttention, GRUAttention, …

## How to start using tsai?

To get to know the tsai package, we’d suggest you start with this notebook in Google Colab: **01_Intro_to_Time_Series_Classification** It provides an overview of a time series classification task.

We have also develop many other tutorial notebooks.

To use tsai in your own notebooks, the only thing you need to do after you have installed the package is to run this:
    
    
    from tsai.all import *__

## Examples

These are just a few examples of how you can use `tsai`:

### Binary, univariate classification

**Training:**
    
    
    from tsai.basics import *
    
    X, y, splits = get_classification_data('ECG200', split_data=False)
    tfms = [None, TSClassification()]
    batch_tfms = TSStandardize()
    clf = TSClassifier(X, y, splits=splits, path='models', arch="InceptionTimePlus", tfms=tfms, batch_tfms=batch_tfms, metrics=accuracy, cbs=ShowGraph())
    clf.fit_one_cycle(100, 3e-4)
    clf.export("clf.pkl") __

**Inference:**
    
    
    from tsai.inference import load_learner
    
    clf = load_learner("models/clf.pkl")
    probas, target, preds = clf.get_X_preds(X[splits[1]], y[splits[1]])__

### Multi-class, multivariate classification

**Training:**
    
    
    from tsai.basics import *
    
    X, y, splits = get_classification_data('LSST', split_data=False)
    tfms = [None, TSClassification()]
    batch_tfms = TSStandardize(by_sample=True)
    mv_clf = TSClassifier(X, y, splits=splits, path='models', arch="InceptionTimePlus", tfms=tfms, batch_tfms=batch_tfms, metrics=accuracy, cbs=ShowGraph())
    mv_clf.fit_one_cycle(10, 1e-2)
    mv_clf.export("mv_clf.pkl")__

**Inference:**
    
    
    from tsai.inference import load_learner
    
    mv_clf = load_learner("models/mv_clf.pkl")
    probas, target, preds = mv_clf.get_X_preds(X[splits[1]], y[splits[1]])__

### Multivariate Regression

**Training:**
    
    
    from tsai.basics import *
    
    X, y, splits = get_regression_data('AppliancesEnergy', split_data=False)
    tfms = [None, TSRegression()]
    batch_tfms = TSStandardize(by_sample=True)
    reg = TSRegressor(X, y, splits=splits, path='models', arch="TSTPlus", tfms=tfms, batch_tfms=batch_tfms, metrics=rmse, cbs=ShowGraph(), verbose=True)
    reg.fit_one_cycle(100, 3e-4)
    reg.export("reg.pkl")__

**Inference:**
    
    
    from tsai.inference import load_learner
    
    reg = load_learner("models/reg.pkl")
    raw_preds, target, preds = reg.get_X_preds(X[splits[1]], y[splits[1]])__

The ROCKETs (RocketClassifier, RocketRegressor, MiniRocketClassifier, MiniRocketRegressor, MiniRocketVotingClassifier or MiniRocketVotingRegressor) are somewhat different models. They are not actually deep learning models (although they use convolutions) and are used in a different way.

⚠️ You’ll also need to install sktime to be able to use them. You can install it separately:
    
    
    pip install sktime __

or use:
    
    
    pip install tsai[extras]__

**Training:**
    
    
    from sklearn.metrics import mean_squared_error, make_scorer
    from tsai.data.external import get_Monash_regression_data
    from tsai.models.MINIROCKET import MiniRocketRegressor
    
    X_train, y_train, *_ = get_Monash_regression_data('AppliancesEnergy')
    rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    reg = MiniRocketRegressor(scoring=rmse_scorer)
    reg.fit(X_train, y_train)
    reg.save('MiniRocketRegressor')__

**Inference:**
    
    
    from sklearn.metrics import mean_squared_error
    from tsai.data.external import get_Monash_regression_data
    from tsai.models.MINIROCKET import load_minirocket
    
    *_, X_test, y_test = get_Monash_regression_data('AppliancesEnergy')
    reg = load_minirocket('MiniRocketRegressor')
    y_pred = reg.predict(X_test)
    mean_squared_error(y_test, y_pred, squared=False)__

### Forecasting

You can use tsai for forecast in the following scenarios:

  * univariate or multivariate time series input
  * univariate or multivariate time series output
  * single or multi-step ahead



You’ll need to: * prepare X (time series input) and the target y (see documentation) * select PatchTST or one of tsai’s models ending in Plus (TSTPlus, InceptionTimePlus, TSiTPlus, etc). The model will auto-configure a head to yield an output with the same shape as the target input y.

#### Single step

**Training:**
    
    
    from tsai.basics import *
    
    ts = get_forecasting_time_series("Sunspots").values
    X, y = SlidingWindow(60, horizon=1)(ts)
    splits = TimeSplitter(235)(y) 
    tfms = [None, TSForecasting()]
    batch_tfms = TSStandardize()
    fcst = TSForecaster(X, y, splits=splits, path='models', tfms=tfms, batch_tfms=batch_tfms, bs=512, arch="TSTPlus", metrics=mae, cbs=ShowGraph())
    fcst.fit_one_cycle(50, 1e-3)
    fcst.export("fcst.pkl")__

**Inference:**
    
    
    from tsai.inference import load_learner
    
    fcst = load_learner("models/fcst.pkl", cpu=False)
    raw_preds, target, preds = fcst.get_X_preds(X[splits[1]], y[splits[1]])
    raw_preds.shape
    # torch.Size([235, 1])__

#### Multi-step

This example show how to build a 3-step ahead univariate forecast.

**Training:**
    
    
    from tsai.basics import *
    
    ts = get_forecasting_time_series("Sunspots").values
    X, y = SlidingWindow(60, horizon=3)(ts)
    splits = TimeSplitter(235, fcst_horizon=3)(y) 
    tfms = [None, TSForecasting()]
    batch_tfms = TSStandardize()
    fcst = TSForecaster(X, y, splits=splits, path='models', tfms=tfms, batch_tfms=batch_tfms, bs=512, arch="TSTPlus", metrics=mae, cbs=ShowGraph())
    fcst.fit_one_cycle(50, 1e-3)
    fcst.export("fcst.pkl")__

**Inference:**
    
    
    from tsai.inference import load_learner
    fcst = load_learner("models/fcst.pkl", cpu=False)
    raw_preds, target, preds = fcst.get_X_preds(X[splits[1]], y[splits[1]])
    raw_preds.shape
    # torch.Size([235, 3])__

## Input data format

The input format for all time series models and image models in tsai is the same. An np.ndarray (or array-like object like zarr, etc) with 3 dimensions:

**[# samples x # variables x sequence length]**

The input format for tabular models in tsai (like TabModel, TabTransformer and TabFusionTransformer) is a pandas dataframe. See example.

## How to contribute to tsai?

We welcome contributions of all kinds. Development of enhancements, bug fixes, documentation, tutorial notebooks, …

We have created a guide to help you start contributing to tsai. You can read it here.

## Enterprise support and consulting services:

Want to make the most out of timeseriesAI/tsai in a professional setting? Let us help. Send us an email to learn more: info@timeseriesai.co

## Citing tsai

If you use tsai in your research please use the following BibTeX entry:
    
    
    @Misc{tsai,
        author =       {Ignacio Oguiza},
        title =        {tsai - A state-of-the-art deep learning library for time series and sequential data},
        howpublished = {Github},
        year =         {2023},
        url =          {https://github.com/timeseriesAI/tsai}
    }

  * __Report an issue


