## On this page

  * MiniRocketClassifier
  * load_minirocket
  * MiniRocketRegressor
  * load_minirocket
  * MiniRocketVotingClassifier
  * get_minirocket_preds
  * MiniRocketVotingRegressor



  * __Report an issue



  1. Models
  2. ROCKETs
  3. MINIROCKET



# MINIROCKET

> A Very Fast (Almost) Deterministic Transform for Time Series Classification.

* * *

source

### MiniRocketClassifier

> 
>      MiniRocketClassifier (num_features=10000, max_dilations_per_kernel=32,
>                            random_state=None, alphas=array([1.e-03, 1.e-02,
>                            1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03]),
>                            normalize_features=True, memory=None,
>                            verbose=False, scoring=None, class_weight=None,
>                            **kwargs)

_Time series classification using MINIROCKET features and a linear classifier_

* * *

source

### load_minirocket

> 
>      load_minirocket (fname, path='./models')

* * *

source

### MiniRocketRegressor

> 
>      MiniRocketRegressor (num_features=10000, max_dilations_per_kernel=32,
>                           random_state=None, alphas=array([1.e-03, 1.e-02,
>                           1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03]),
>                           normalize_features=True, memory=None, verbose=False,
>                           scoring=None, **kwargs)

_Time series regression using MINIROCKET features and a linear regressor_

* * *

source

### load_minirocket

> 
>      load_minirocket (fname, path='./models')

* * *

source

### MiniRocketVotingClassifier

> 
>      MiniRocketVotingClassifier (n_estimators=5, weights=None, n_jobs=-1,
>                                  num_features=10000,
>                                  max_dilations_per_kernel=32,
>                                  random_state=None, alphas=array([1.e-03,
>                                  1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02,
>                                  1.e+03]), normalize_features=True,
>                                  memory=None, verbose=False, scoring=None,
>                                  class_weight=None, **kwargs)

_Time series classification ensemble using MINIROCKET features, a linear classifier and majority voting_

* * *

source

### get_minirocket_preds

> 
>      get_minirocket_preds (X, fname, path='./models', model=None)

* * *

source

### MiniRocketVotingRegressor

> 
>      MiniRocketVotingRegressor (n_estimators=5, weights=None, n_jobs=-1,
>                                 num_features=10000,
>                                 max_dilations_per_kernel=32,
>                                 random_state=None, alphas=array([1.e-03,
>                                 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02,
>                                 1.e+03]), normalize_features=True,
>                                 memory=None, verbose=False, scoring=None,
>                                 **kwargs)

_Time series regression ensemble using MINIROCKET features, a linear regressor and a voting regressor_
    
    
    # Univariate classification with sklearn-type API
    dsid = 'OliveOil'
    fname = 'MiniRocketClassifier'
    X_train, y_train, X_test, y_test = get_UCR_data(dsid)
    cls = MiniRocketClassifier()
    cls.fit(X_train, y_train)
    cls.save(fname)
    pred = cls.score(X_test, y_test)
    del cls
    cls = load_minirocket(fname)
    test_eq(cls.score(X_test, y_test), pred)__
    
    
    OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
    
    
    # Multivariate classification with sklearn-type API
    dsid = 'NATOPS'
    X_train, y_train, X_test, y_test = get_UCR_data(dsid)
    cls = MiniRocketClassifier()
    cls.fit(X_train, y_train)
    cls.score(X_test, y_test)__
    
    
    0.9277777777777778
    
    
    # Multivariate classification with sklearn-type API
    dsid = 'NATOPS'
    X_train, y_train, X_test, y_test = get_UCR_data(dsid)
    cls = MiniRocketVotingClassifier(5)
    cls.fit(X_train, y_train)
    cls.score(X_test, y_test)__
    
    
    OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
    OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
    OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
    OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
    OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
    
    
    0.9166666666666666
    
    
    from sklearn.metrics import mean_squared_error __
    
    
    # Univariate regression with sklearn-type API
    dsid = 'Covid3Month'
    fname = 'MiniRocketRegressor'
    X_train, y_train, X_test, y_test = get_Monash_regression_data(dsid)
    if X_train is not None:
        rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
        reg = MiniRocketRegressor(scoring=rmse_scorer)
        reg.fit(X_train, y_train)
        reg.save(fname)
        del reg
        reg = load_minirocket(fname)
        y_pred = reg.predict(X_test)
        print(mean_squared_error(y_test, y_pred, squared=False))__
    
    
    0.04099244037606886
    
    
    # Multivariate regression with sklearn-type API
    dsid = 'AppliancesEnergy'
    X_train, y_train, X_test, y_test = get_Monash_regression_data(dsid)
    if X_train is not None:
        rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
        reg = MiniRocketRegressor(scoring=rmse_scorer)
        reg.fit(X_train, y_train)
        reg.save(fname)
        del reg
        reg = load_minirocket(fname)
        y_pred = reg.predict(X_test)
        print(mean_squared_error(y_test, y_pred, squared=False))__
    
    
    2.2938026879322577
    
    
    # Multivariate regression ensemble with sklearn-type API
    if X_train is not None:
        reg = MiniRocketVotingRegressor(5, scoring=rmse_scorer)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print(mean_squared_error(y_test, y_pred, squared=False))__
    
    
    OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
    OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
    
    
    2.286295546348893

  * __Report an issue


