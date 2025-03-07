## On this page

  * RocketClassifier
  * load_rocket
  * RocketRegressor



  * __Report an issue



  1. Models
  2. ROCKETs
  3. ROCKET



# ROCKET

> ROCKET (RandOm Convolutional KErnel Transform) functions for univariate and multivariate time series.

* * *

source

### RocketClassifier

> 
>      RocketClassifier (num_kernels=10000, normalize_input=True,
>                        random_state=None, alphas=array([1.e-03, 1.e-02,
>                        1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03]),
>                        normalize_features=True, memory=None, verbose=False,
>                        scoring=None, class_weight=None, **kwargs)

_Time series classification using ROCKET features and a linear classifier_

* * *

source

### load_rocket

> 
>      load_rocket (fname='Rocket', path='./models')

* * *

source

### RocketRegressor

> 
>      RocketRegressor (num_kernels=10000, normalize_input=True,
>                       random_state=None, alphas=array([1.e-03, 1.e-02, 1.e-01,
>                       1.e+00, 1.e+01, 1.e+02, 1.e+03]),
>                       normalize_features=True, memory=None, verbose=False,
>                       scoring=None, **kwargs)

_Time series regression using ROCKET features and a linear regressor_
    
    
    # Univariate classification with sklearn-type API
    dsid = 'OliveOil'
    fname = 'RocketClassifier'
    X_train, y_train, X_test, y_test = get_UCR_data(dsid, Xdtype='float64')
    cls = RocketClassifier()
    cls.fit(X_train, y_train)
    cls.save(fname)
    del cls
    cls = load_rocket(fname)
    print(cls.score(X_test, y_test))__
    
    
    OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
    
    
    0.9
    
    
    # Multivariate classification with sklearn-type API
    dsid = 'NATOPS'
    fname = 'RocketClassifier'
    X_train, y_train, X_test, y_test = get_UCR_data(dsid, Xdtype='float64')
    cls = RocketClassifier()
    cls.fit(X_train, y_train)
    cls.save(fname)
    del cls
    cls = load_rocket(fname)
    print(cls.score(X_test, y_test))__
    
    
    0.8666666666666667
    
    
    from sklearn.metrics import mean_squared_error __
    
    
    # Univariate regression with sklearn-type API
    dsid = 'Covid3Month'
    fname = 'RocketRegressor'
    X_train, y_train, X_test, y_test = get_Monash_regression_data(dsid, Xdtype='float64')
    if X_train is not None: 
        rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
        reg = RocketRegressor(scoring=rmse_scorer)
        reg.fit(X_train, y_train)
        reg.save(fname)
        del reg
        reg = load_rocket(fname)
        y_pred = reg.predict(X_test)
        print(mean_squared_error(y_test, y_pred, squared=False))__
    
    
    0.03908714523468997
    
    
    # Multivariate regression with sklearn-type API
    dsid = 'AppliancesEnergy'
    fname = 'RocketRegressor'
    X_train, y_train, X_test, y_test = get_Monash_regression_data(dsid, Xdtype='float64')
    if X_train is not None: 
        rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
        reg = RocketRegressor(scoring=rmse_scorer)
        reg.fit(X_train, y_train)
        reg.save(fname)
        del reg
        reg = load_rocket(fname)
        y_pred = reg.predict(X_test)
        print(mean_squared_error(y_test, y_pred, squared=False))__
    
    
    2.287302226812576

  * __Report an issue


