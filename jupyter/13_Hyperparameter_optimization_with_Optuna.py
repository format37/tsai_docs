#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/dnth/tsai/blob/master/tutorial_nbs/11_Optuna_HPO.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# created by Dickson Neoh - dickson.neoh@gmail.com

# ## Purpose 😇

# The purpose of this notebook is to show you how you can take any model or dataset in TSAI and run a hyperparameter optimization job to search for optimal hyperparameter combination that yields the best result on the dataset.

# We'll use [Optuna](https://optuna.readthedocs.io/en/stable/index.html) which is an automatic hyperparameter optimization software framework, particularly designed for machine learning.

# ## Import libraries 📚

# Optuna is not a `tsai` dependency, and needs to be installed separately.

# In[ ]:


# # **************** UNCOMMENT AND RUN THIS CELL IF YOU NEED TO INSTALL/ UPGRADE TSAI ****************
# stable = True # Set to True for latest pip version or False for main branch in GitHub
# !pip install {"tsai -U" if stable else "git+https://github.com/timeseriesAI/tsai.git"} >> /dev/null
# !pip install optuna -U >> /dev/null


# In[ ]:


from tsai.all import *
import optuna
my_setup(optuna)


# In[ ]:


# Sets random seed for random, torch, and numpy (where available)
set_seed(77, False)


# ## Baseline 📉

# Before embarking on any hyperparameter optimization tasks, it is important to get a baseline performance so that we can note the improvements after the optimization is done.
# In this notebook we use the InceptionTimePlus model and and train on the NATOPS dataset, both conveniently provided in TSAI in just few lines of codes. 
# Feel free to use any other models and datasets.

# In[ ]:


X, y, splits = get_UCR_data('NATOPS', split_data=False)
learn = TSClassifier(X, y, splits=splits, bs=[64, 128], batch_tfms=TSStandardize(by_sample=True), arch=InceptionTimePlus, metrics=accuracy, cbs=ShowGraph())
lr_max = learn.lr_find()
print(f'lf_max = {lr_max.valley}')


# In[ ]:


learn = TSClassifier(X, y, splits=splits, bs=[64, 128], batch_tfms=TSStandardize(by_sample=True), arch=InceptionTimePlus, metrics=accuracy, cbs=ShowGraph())
learn.fit_one_cycle(5, lr_max=lr_max.valley)


# Note the performance of the baseline model. It is usually <50% accuracy on my local machine.

# ## Define objective function 🎯

# To define any study in Optuna, you need to create an objective function that will be optimized. The function will be different per study, but it's pretty easy to build with tsai. Below you can find an objective function that you can adapt to meet your needs.
# 
# There are two components in the objective function that you need to define:
# 1. Search space - the hyperparameter values that you would like to search. In this example we are searching for the combination of nf and dropout rate.
# 2. Objective value - the value that will be used to indicate the performance of the model. In this example we use the validation loss as the objective value.
# 
# FastAIPruningCallback is a callback that can optionally be used to prune unpromising trials (ie early stopping) in Optuna. You can choose if you want to monitor the 'valid_loss' or any metric of your choice.

# In[ ]:


import optuna
from optuna.integration import FastAIPruningCallback

def objective(trial:optuna.Trial):
    
    # Define search space here. More info here https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html
    nf = trial.suggest_categorical('num_filters', [32, 64, 96]) # search through all categorical values in the provided list
    depth = trial.suggest_int('depth', 3, 9, step=3) # search through all integer values between 3 and 9 with 3 increment steps
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5, step=.1) # search through all float values between 0.0 and 0.5 with 0.1 increment steps
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)  # search through all float values between 0.0 and 0.5 in log increment steps
    
    batch_tfms = TSStandardize(by_sample=True)
    learn = TSClassifier(X, y, splits=splits, bs=[64, 128], batch_tfms=batch_tfms,
                         arch=InceptionTimePlus, arch_config={'nf':nf, 'fc_dropout':dropout_rate, 'depth':depth},
                         metrics=accuracy, cbs=FastAIPruningCallback(trial))


    with ContextManagers([learn.no_logging(), learn.no_bar()]): # [Optional] this prevents fastai from printing anything during training
        learn.fit_one_cycle(5, lr_max=learning_rate)

    # Return the objective value
    return learn.recorder.values[-1][1] # return the validation loss value of the last epoch 


# ## Start the study 🧑‍🎓

# In Optuna, the hyperparameter search job is known as a Study. Each Study consists of many Trials. The number of Trials indicate how many times do you want Optuna to search through the search space. After configuring the objective function above we would like to let Optuna to perform the search (study) for the combination of hyperparameters that yield the best objective value. 
# 
# 📝Note: In the objective function we used the validation loss as our our objective value. Hence in the the study, we must tell Optuna minimize the objective value (This can be specified in the `direction='minimize'` as shown below). Alternatively, if you have chosen to use the accuracy metric as the objective value, you will need to tell Optuna to maximize instead. (This can be specified in the `direction='maximize'` )

# In[ ]:


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)


# In[ ]:


print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# For reference, the default hyperparameter for the baseline `InceptionTimePlus` [model]("https://github.com/timeseriesAI/tsai/blob/main/tsai/models/InceptionTimePlus.py") is `nf=32`, `depth=6`, and `dropout_rate=0.0`

# In[ ]:


display(optuna.visualization.plot_optimization_history(study))
display(optuna.visualization.plot_param_importances(study))
display(optuna.visualization.plot_slice(study))
display(optuna.visualization.plot_parallel_coordinate(study))


# ## Retrain the model with best hyperparameters

# Now that we have obtained the optimized hyperparameters from the Optuna study, we can train the InceptionTimePlus model again with the optimal hyperparameter values and note the improvement from the baseline model.

# In[ ]:


# Get the best nf and dropout rate from the best trial object
trial = study.best_trial
nf = trial.params['num_filters']
depth = trial.params['depth']
dropout_rate = trial.params['dropout_rate']
learning_rate = trial.params['learning_rate']


# In[ ]:


learn = TSClassifier(X, y, splits=splits, bs=[64, 128], 
                     batch_tfms=TSStandardize(by_sample=True), 
                     arch=InceptionTimePlus, arch_config={'nf':nf, 'fc_dropout':dropout_rate, 'depth':depth},
                     metrics=accuracy, cbs=ShowGraph())
learn.fit_one_cycle(5, lr_max=learning_rate)


# For comparison our baseline model can only scored <50% accuracy with 5 epochs training (This might vary on your machine). However, using the hyperparameters from the Optuna study results in much higher accuracy. The numbers might vary due to the randomness in training. You can seed you runs or train the model a few times to verify the results. Sometimes the study fails to find a combination that works better than the baseline. In this case you might want to increase the number of trials in the study.
# 
# Happy learning! 

# In[ ]:




