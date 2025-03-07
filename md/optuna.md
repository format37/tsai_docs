## On this page

  * run_optuna_study



  * __Report an issue



  1. HPO & experiment tracking
  2. Optuna



# Optuna

> A hyperparameter optimization framework

Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It features an imperative, define-by-run style user API. Thanks to our define-by-run API, the code written with Optuna enjoys high modularity, and the user of Optuna can dynamically construct the search spaces for the hyperparameters.

* * *

source

### run_optuna_study

> 
>      run_optuna_study (objective, resume=None, study_type=None,
>                        multivariate=True, search_space=None, evaluate=None,
>                        seed=None, sampler=None, pruner=None, study_name=None,
>                        direction='maximize', n_trials=None, timeout=None,
>                        gc_after_trial=False, show_progress_bar=True,
>                        save_study=True, path='optuna', show_plots=True)

*Creates and runs an optuna study.

Args: objective: A callable that implements objective function. resume: Path to a previously saved study. study_type: Type of study selected (bayesian, gridsearch, randomsearch). Based on this a sampler will be build if sampler is None. If a sampler is passed, this has no effect. multivariate: If this is True, the multivariate TPE is used when suggesting parameters. The multivariate TPE is reported to outperform the independent TPE. search_space: Search space required when running a gridsearch (if you don’t pass a sampler). evaluate: Allows you to pass a specific set of hyperparameters that will be evaluated. seed: Fixed seed used by samplers. sampler: A sampler object that implements background algorithm for value suggestion. If None is specified, TPESampler is used during single-objective optimization and NSGAIISampler during multi-objective optimization. See also samplers. pruner: A pruner object that decides early stopping of unpromising trials. If None is specified, MedianPruner is used as the default. See also pruners. study_name: Study’s name. If this argument is set to None, a unique name is generated automatically. direction: A sequence of directions during multi-objective optimization. n_trials: The number of trials. If this argument is set to None, there is no limitation on the number of trials. If timeout is also set to None, the study continues to create trials until it receives a termination signal such as Ctrl+C or SIGTERM. timeout: Stop study after the given number of second(s). If this argument is set to None, the study is executed without time limitation. If n_trials is also set to None, the study continues to create trials until it receives a termination signal such as Ctrl+C or SIGTERM. gc_after_trial: Flag to execute garbage collection at the end of each trial. By default, garbage collection is enabled, just in case. You can turn it off with this argument if memory is safely managed in your objective function. show_progress_bar: Flag to show progress bars or not. To disable progress bar, set this False. save_study: Save your study when finished/ interrupted. path: Folder where the study will be saved. show_plots: Flag to control whether plots are shown at the end of the study.*

Exported source
    
    
    def run_optuna_study(objective, resume=None, study_type=None, multivariate=True, search_space=None, evaluate=None, seed=None, sampler=None, pruner=None, 
                         study_name=None, direction='maximize', n_trials=None, timeout=None, gc_after_trial=False, show_progress_bar=True, 
                         save_study=True, path='optuna', show_plots=True):
        r"""Creates and runs an optuna study.
    
        Args: 
            objective:          A callable that implements objective function.
            resume:             Path to a previously saved study.
            study_type:         Type of study selected (bayesian, gridsearch, randomsearch). Based on this a sampler will be build if sampler is None. 
                                If a sampler is passed, this has no effect.
            multivariate:       If this is True, the multivariate TPE is used when suggesting parameters. The multivariate TPE is reported to outperform 
                                the independent TPE.
            search_space:       Search space required when running a gridsearch (if you don't pass a sampler).
            evaluate:           Allows you to pass a specific set of hyperparameters that will be evaluated.
            seed:               Fixed seed used by samplers.
            sampler:            A sampler object that implements background algorithm for value suggestion. If None is specified, TPESampler is used during 
                                single-objective optimization and NSGAIISampler during multi-objective optimization. See also samplers.
            pruner:             A pruner object that decides early stopping of unpromising trials. If None is specified, MedianPruner is used as the default. 
                                See also pruners.
            study_name:         Study’s name. If this argument is set to None, a unique name is generated automatically.
            direction:          A sequence of directions during multi-objective optimization.
            n_trials:           The number of trials. If this argument is set to None, there is no limitation on the number of trials. If timeout is also set to 
                                None, the study continues to create trials until it receives a termination signal such as Ctrl+C or SIGTERM.
            timeout:            Stop study after the given number of second(s). If this argument is set to None, the study is executed without time limitation. 
                                If n_trials is also set to None, the study continues to create trials until it receives a termination signal such as 
                                Ctrl+C or SIGTERM.
            gc_after_trial:     Flag to execute garbage collection at the end of each trial. By default, garbage collection is enabled, just in case. 
                                You can turn it off with this argument if memory is safely managed in your objective function.
            show_progress_bar:  Flag to show progress bars or not. To disable progress bar, set this False.
            save_study:         Save your study when finished/ interrupted.
            path:               Folder where the study will be saved.
            show_plots:         Flag to control whether plots are shown at the end of the study.
        """
        
        try: import optuna
        except ImportError: raise ImportError('You need to install optuna to use run_optuna_study')
    
        # Sampler
        if sampler is None:
            if study_type is None or "bayes" in study_type.lower(): 
                sampler = optuna.samplers.TPESampler(seed=seed, multivariate=multivariate)
            elif "grid" in study_type.lower():
                assert search_space, f"you need to pass a search_space dict to run a gridsearch"
                sampler = optuna.samplers.GridSampler(search_space)
            elif "random" in study_type.lower(): 
                sampler = optuna.samplers.RandomSampler(seed=seed)
        assert sampler, "you need to either select a study type (bayesian, gridsampler, randomsampler) or pass a sampler"
    
        # Study
        if resume: 
            try:
                study = joblib.load(resume)
            except: 
                print(f"joblib.load({resume}) couldn't recover any saved study. Check the path.")
                return
            print("Best trial until now:")
            print(" Value: ", study.best_trial.value)
            print(" Params: ")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value}")
        else: 
            study = optuna.create_study(sampler=sampler, pruner=pruner, study_name=study_name, direction=direction)
        if evaluate: study.enqueue_trial(evaluate)
        try:
            study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=gc_after_trial, show_progress_bar=show_progress_bar)
        except KeyboardInterrupt:
            pass
    
        # Save
        if save_study:
            full_path = Path(path)/f'{study.study_name}.pkl'
            full_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(study, full_path)
            print(f'\nOptuna study saved to {full_path}')
            print(f"To reload the study run: study = joblib.load('{full_path}')")
    
        # Plots
        if show_plots and len(study.trials) > 1:
            try: display(optuna.visualization.plot_optimization_history(study))
            except: pass
            try: display(optuna.visualization.plot_param_importances(study))
            except: pass
            try: display(optuna.visualization.plot_slice(study))
            except: pass
            try: display(optuna.visualization.plot_parallel_coordinate(study))
            except: pass
    
        # Study stats
        try:
            pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            print(f"\nStudy statistics    : ")
            print(f"  Study name        : {study.study_name}")
            print(f"  # finished trials : {len(study.trials)}")
            print(f"  # pruned trials   : {len(pruned_trials)}")
            print(f"  # complete trials : {len(complete_trials)}")
            
            print(f"\nBest trial          :")
            trial = study.best_trial
            print(f"  value             : {trial.value}")
            print(f"  best_params = {trial.params}\n")
        except:
            print('\nNo finished trials yet.')
        return study __

  * __Report an issue


