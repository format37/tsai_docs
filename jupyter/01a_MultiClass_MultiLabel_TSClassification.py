#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/01_Intro_to_Time_Series_Classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# created by Doug Williams (https://github.com/williamsdoug) and Ignacio Oguiza (https://github.com/timeseriesAI/tsai). 
# 
# contact: oguiza@timeseriesAI.co

# ## Purpose 😇

# The purpose of this notebook is to demonstrate both multi-class and multi-label classification using `tsai`. 
# 
# - Multi-class classification: While the output can take on multiple possible values, for any given sample the output can take on only a single value. In other words, the label values are mutually exclusive. Implication are:
#   - CategoricalCrossEntropy (`nn.CrossEntropyLoss` in Pytorch, `CrossEntropyLossFlat` in fastai) is used as the loss function during training
#   - Softmax is used to determine prediction since only one label value can be true, the predicted label is the value label value with the greatest probability. 
#   - Softmax reduces the potential for spurious label predictions since only the label with the highest probability is selected
#   - In both Pytorch and fastai the loss combines a Softmax layer and the CrossEntropyLoss in one single class, so Softmax shouldn't be added to the model.
#  
# - Multi-label classification: This is the more general case where an individual sample can have one or more labels, relaxing the mutual label exclusivity constraint. Implications are:
#   - BinaryCrossEntropy (`nn.BCEWithLogitsLoss` in Pytorch, `BCEWithLogitsLossFlat` in fastai) is used as the loss function during training
#   - Sigmoid is used to determine prediction since multiple labels may be true. In both Pytorch and fastai the loss combines a Sigmoid layer and the BCELoss in one single class, so Sigmoid shouldn't be added to the model.
#   - Relative to multi-class classification, multi-label classification may be more prone to spurious false-positive labels.

# ## Import libraries 📚

# In[ ]:


# # **************** UNCOMMENT AND RUN THIS CELL IF YOU NEED TO INSTALL/ UPGRADE TSAI ****************
# stable = True # Set to True for latest pip version or False for main branch in GitHub
# !pip install {"tsai -U" if stable else "git+https://github.com/timeseriesAI/tsai.git"} >> /dev/null


# In[ ]:


from tsai.all import *
my_setup()


# ## Prepare data 🔢
# 
# For this example we will be using the UCR ECG5000 heartbeat dataset with is based on the [Physionet BIDMC Congestive Heart Failure Database](https://physionet.org/content/chfdb/1.0.0/]), specifically recording chf07.
# 
# For the purposes of this example the UCR labels will be mapped to more descriptive labels:
# 
# - 1 - Normal ('Nor')
# - 2 - R-on-T premature ventricular contraction ('RoT')
# - 3 - Premature ventricular contraction ('PVC')
# - 4 - Supraventricular premature or ectopic beat, atrial or nodal ('SPC')
# - 5 - Unclassifiable beat ('Unk')

# In[ ]:


class_map = {
    '1':'Nor',  # N:1  - Normal
    '2':'RoT',  # r:2  - R-on-T premature ventricular contraction
    '3':'PVC',  # V:3  - Premature ventricular contraction
    '4':'SPC',  # S:4  - Supraventricular premature or ectopic beat (atrial or nodal)
    '5':'Unk',  # Q:5  - Unclassifiable beat
    }
class_map


# In[ ]:


# dataset id
dsid = 'ECG5000' 
X, y, splits = get_UCR_data(dsid, split_data=False)
labeler = ReLabeler(class_map)
new_y = labeler(y) # map to more descriptive labels
X.shape, new_y.shape, splits, new_y


# In[ ]:


label_counts = collections.Counter(new_y)
print('Counts by label:', dict(label_counts))
print(f'Naive Accuracy: {100*max(label_counts.values())/sum(label_counts.values()):0.2f}%')


# Note: naive accuracy is calculated by assuming all samples are predicted as 'Nor' (most frequent label).

# ## Multi-class classification

# ### Prepare dataloaders 💿

# In[ ]:


tfms  = [None, TSClassification()] # TSClassification == Categorize
batch_tfms = TSStandardize()
dls = get_ts_dls(X, new_y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=[64, 128])
dls.dataset


# ### Visualize data

# In[ ]:


dls.show_batch(sharey=True)


# ### Build learner 🏗

# There are at least 2 equivalent ways to build a time series learner: 

# In[ ]:


model = build_ts_model(InceptionTimePlus, dls=dls)
learn = Learner(dls, model, metrics=accuracy)


# In[ ]:


learn = ts_learner(dls, metrics=accuracy) # == ts_learner(dls, arch=InceptionTimePlus, metrics=accuracy) since InceptionTimePlus is the default arch


# ### LR find 🔎

# In[ ]:


learn.lr_find()


# ### Train 🏃🏽‍♀️

# In[ ]:


learn = ts_learner(dls, metrics=accuracy, cbs=ShowGraph())
learn.fit_one_cycle(10, lr_max=1e-3)


# Now that the model is trained, we'll save it to create predictions in the future: 

# In[ ]:


PATH = Path('./models/Multiclass.pkl')
PATH.parent.mkdir(parents=True, exist_ok=True)
learn.export(PATH)


# ### Visualize results 👁

# In[ ]:


learn.show_results(sharey=True)


# In[ ]:


learn.show_probas()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[ ]:


interp.most_confused(min_val=3)


# ### Create predictions 🌦

# To get predictions we need to create a learner object from the saved file: 

# In[ ]:


PATH = Path('./models/Multiclass.pkl')
learn_gpu = load_learner(PATH, cpu=False)


# We can now generate predictions using just the X as input. This can be done with a 'gpu' or a 'cpu'. And for many samples, or just one at a time: 

# In[ ]:


# gpu, many samples
probas, _, preds = learn_gpu.get_X_preds(X[splits[1]])
preds[-10:]


# In[ ]:


skm.accuracy_score(new_y[splits[1]], preds)


# In[ ]:


PATH = Path('./models/Multiclass.pkl')
learn_cpu = load_learner(PATH, cpu=True)


# In[ ]:


# cpu, single sample
probas, _, preds = learn_cpu.get_X_preds(X[-1][None]) # [None] is added to pass a 3D array with dimensions [batch_size x n_vars x seq_len]
preds


# ## Multi-label Classification

# ### Augment labels to demonstrate multi-label
# 
# - Create additional label premature ('Pre')
# - Include with any sample where labels 'RoT','PVC','SPC' are already present
# 
# 
# Note:  While in this example the new Pre label is a composite of existing labels, more typically multi-label classification problems include orthogonal label groups.  For example in ECG classification one might have labels related to timing (e.g.: premature), QRS shape (e.g.: block) and other factors (e.g.: ST elevation or depression)

# In[ ]:


class_map = {
    '1':['Nor'],          # N:1  - Normal
    '2':['RoT', 'Pre'],   # r:2  - R-on-T premature ventricular contraction
    '3':['PVC', 'Pre'] ,  # V:3  - Premature ventricular contraction
    '4':['SPC', 'Pre'],   # S:4  - Supraventricular premature or ectopic beat (atrial or nodal)
    '5':['Unk'],          # Q:5  - Unclassifiable beat
    }
class_map


# In[ ]:


labeler = ReLabeler(class_map)
y_multi = labeler(y)
y_multi


# In[ ]:


label_counts = collections.Counter([a for r in y_multi for a in r])
print('Counts by label:', dict(label_counts))


# ### Prepare Dataloaders
# 
# - Replace earlier ```tfms  = [None, [TSClassification()]]```  with ```tfms  = [None, TSMultiLabelClassification()]```
# - TSMultiLabelClassification() is equivalent to [MultiCategorize(), OneHotEncode()] combined in a single transform.
# - When creating dataloaders in multi-label problems, always leave inplace=True (default value) to avoid issues.

# In[ ]:


tfms  = [None, TSMultiLabelClassification()] # TSMultiLabelClassification() == [MultiCategorize(), OneHotEncode()]
batch_tfms = TSStandardize()
dls = get_ts_dls(X, y_multi, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=[64, 128])
dls.dataset


# In[ ]:


dls.show_batch(sharey=True)


# ### Build Learner and Train
# 
# - Use ```metrics=accuracy_multi``` in place of ```metrics=accuracy``` used in earlier multi-class example
# 
# - accuracy_multi can be calculated by sample (default) or by predicted label. When using accuracy_multi by_sample=True, all predicted labels for each sample must be correct to be counted as a correct sample. This is sometimes difficult (when having too many labels). There are cases when labels are partially correct. If we want to account for these, we'll set by_sample to False. 
# 
# - when using amulti-label dataset, we need to choose a model is tsai ending in Plus. These models are essentially the same as the ones with the suffix Plus, but provide some additional flexibility. In our case we'll use ```InceptionTimePlus```. If you don't pass any architecture ```InceptionTimePlus``` will be use. When in doubt, use ```InceptionTimePlus```.

# In[ ]:


def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True, by_sample=False):
    "Computes accuracy when `inp` and `targ` are the same size."
    if sigmoid: inp = inp.sigmoid()
    correct = (inp>thresh)==targ.bool()
    if by_sample:
        return (correct.float().mean(-1) == 1).float().mean()
    else:
        inp,targ = flatten_check(inp,targ)
        return correct.float().mean()


# In[ ]:


learn = ts_learner(dls, InceptionTimePlus, metrics=accuracy_multi)
learn.lr_find()


# In[ ]:


learn = ts_learner(dls, InceptionTimePlus, metrics=[partial(accuracy_multi, by_sample=True), partial(accuracy_multi, by_sample=False)], cbs=ShowGraph())
learn.fit_one_cycle(10, lr_max=1e-3)


# ### Additional Multi-label Metrics

# A naive classifier that always predicts no true labels would achieve 76% accuracy, so the classifier should do much better.  This also demonstrates the weakness of relying overly on the accuracy metric (with by_sample=False), due to the prevalance of false outputs.

# In[ ]:


label_counts = collections.Counter([a for r in y_multi for a in r])
print(f'Naive Accuracy: {100*(1-sum(label_counts.values())/(len(y_multi)*len(label_counts))):0.2f}%')


# ### Define Metrics
# 
# We have included a number of multilabel metrics in `tsai` based on definitions in [Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall#F-measure)

# In[ ]:


def precision_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Computes precision when `inp` and `targ` are the same size."
    
    inp,targ = flatten_check(inp,targ)
    if sigmoid: inp = inp.sigmoid()
    pred = inp>thresh
    
    correct = pred==targ.bool()
    TP = torch.logical_and(correct,  (targ==1).bool()).sum()
    FP = torch.logical_and(~correct, (targ==0).bool()).sum()

    precision = TP/(TP+FP)
    return precision

def recall_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Computes recall when `inp` and `targ` are the same size."
    
    inp,targ = flatten_check(inp,targ)
    if sigmoid: inp = inp.sigmoid()
    pred = inp>thresh
    
    correct = pred==targ.bool()
    TP = torch.logical_and(correct,  (targ==1).bool()).sum()
    FN = torch.logical_and(~correct, (targ==1).bool()).sum()

    recall = TP/(TP+FN)
    return recall

def specificity_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Computes specificity (true negative rate) when `inp` and `targ` are the same size."
    
    inp,targ = flatten_check(inp,targ)
    if sigmoid: inp = inp.sigmoid()
    pred = inp>thresh
    
    correct = pred==targ.bool()
    TN = torch.logical_and(correct,  (targ==0).bool()).sum()
    FP = torch.logical_and(~correct, (targ==0).bool()).sum()

    specificity = TN/(TN+FP)
    return specificity

def balanced_accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Computes balanced accuracy when `inp` and `targ` are the same size."
    
    inp,targ = flatten_check(inp,targ)
    if sigmoid: inp = inp.sigmoid()
    pred = inp>thresh
    
    correct = pred==targ.bool()
    TP = torch.logical_and(correct,  (targ==1).bool()).sum()
    TN = torch.logical_and(correct,  (targ==0).bool()).sum()
    FN = torch.logical_and(~correct, (targ==1).bool()).sum()
    FP = torch.logical_and(~correct, (targ==0).bool()).sum()

    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    balanced_accuracy = (TPR+TNR)/2
    return balanced_accuracy

def Fbeta_multi(inp, targ, beta=1.0, thresh=0.5, sigmoid=True):
    "Computes Fbeta when `inp` and `targ` are the same size."
    
    inp,targ = flatten_check(inp,targ)
    if sigmoid: inp = inp.sigmoid()
    pred = inp>thresh
    
    correct = pred==targ.bool()
    TP = torch.logical_and(correct,  (targ==1).bool()).sum()
    TN = torch.logical_and(correct,  (targ==0).bool()).sum()
    FN = torch.logical_and(~correct, (targ==1).bool()).sum()
    FP = torch.logical_and(~correct, (targ==0).bool()).sum()

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    beta2 = beta*beta
    
    if precision+recall > 0:
        Fbeta = (1+beta2)*precision*recall/(beta2*precision+recall)
    else:
        Fbeta = 0
    return Fbeta

def F1_multi(*args, **kwargs):
    return Fbeta_multi(*args, **kwargs)  # beta defaults to 1.0


# ### Train Classifier Including More Metrics

# In[ ]:


metrics =[accuracy_multi, balanced_accuracy_multi, precision_multi, recall_multi, specificity_multi, F1_multi] 
learn = ts_learner(dls, InceptionTimePlus, metrics=metrics, cbs=ShowGraph())
learn.fit_one_cycle(10, lr_max=1e-3)


# ### Optionally Use Class (Positive) Weights
# 
# - Include Per-label positive weights bias the loss function to give greater importance to samples where label is present with loss function
# - `tsai` automatically calculates class weights. It's a dataloaders attribute called cws. **You should use dls.train.cws to avoid any leakage.**

# ### Prepare DataLoaders

# In[ ]:


tfms  = [None, TSMultiLabelClassification()] # TSMultiLabelClassification() == [MultiCategorize(), OneHotEncode()]
batch_tfms = TSStandardize()
dls = get_ts_dls(X, y_multi, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=[64, 128])
dls.vocab, dls.train.cws


# ### Build Learner and Train
# 
# - Include class weights with ```loss_func```

# In[ ]:


metrics = [accuracy_multi, balanced_accuracy_multi, precision_multi, recall_multi,  specificity_multi, F1_multi]
learn = ts_learner(dls, InceptionTimePlus, metrics=metrics, loss_func=BCEWithLogitsLossFlat(pos_weight=dls.train.cws), cbs=ShowGraph())


# In[ ]:


learn.lr_find()


# In[ ]:


learn.fit_one_cycle(20, lr_max=1e-3)


# ### Try Again with Reduced Weights (reduce by sqrt)

# In[ ]:


metrics = [accuracy_multi, balanced_accuracy_multi, precision_multi, recall_multi,  specificity_multi, F1_multi]
learn = ts_learner(dls, InceptionTimePlus, metrics=metrics, loss_func=BCEWithLogitsLossFlat(pos_weight=dls.train.cws.sqrt()), cbs=ShowGraph())


# In[ ]:


learn.lr_find()


# In[ ]:


learn.fit_one_cycle(10, lr_max=1e-3)


# Let's save the learner for inference: 

# In[ ]:


PATH = Path('./models/Multilabel.pkl')
PATH.parent.mkdir(parents=True, exist_ok=True)
learn.export(PATH)


# ### Create predictions 🌦

# To get predictions we need to create a learner object from the saved file: 

# In[ ]:


PATH = Path('./models/Multilabel.pkl')
learn_gpu = load_learner(PATH, cpu=False)


# We can now generate predictions using just the X as input. This can be done with a 'gpu' or a 'cpu'. And for many samples, or just one at a time: 

# In[ ]:


# gpu, many samples, multilabel
probas, _, preds = learn_gpu.get_X_preds(X[splits[1]])
preds[-10:]


# In[ ]:


PATH = Path('./models/Multilabel.pkl')
learn_cpu = load_learner(PATH, cpu=True)


# In[ ]:


# cpu, single sample
probas, _, preds = learn_cpu.get_X_preds(X[-1][None])
preds


# ### Summary of Key Points: Multi-Label Classification
# 
# #### Basic Data Preparation
# - Replace earlier ```tfms  = [None, TSClassification()]```  with ```tfms  = [None, TSMultiLabelClassification()]```
# 
# #### Basic Learner
# 
# - You can build a learnr using ```learn = ts_learner()``` or  ```learn = Learner()```
# - If you choose ```ts_learner``` you can pass an architecture (rather than passing pre-created model) or leave it as None, in which case the default (InceptiontimePlus) will be used.
# - Use ```metrics=accuracy_multi``` in place of ```metrics=accuracy``` used in earlier multi-class example
# - If no loss_func is passed, `tsai` will set it to loss function ```loss_func=BCEWithLogitsLossFlat()```.
#   
# #### Multi-Label Metrics
# - Remember you can use accuracy_multi(by_sample=True) which will consider as correct samples where all labels are correct. If by_sample=False, all labels for all samples will be considered, which may lead to a biased result.
# - Consider using multi=label metrics, such as those included in this example.
# 
# #### Optionally Include Positive Weights with Loss Function
# - Since large number of negative targets after application of binary encoding, label weighting may better optimize loss function for positive labels
# - `tsai` automatically calculates multi class weights. 
# - Remember to use the ones in the train set: dls.train.cws
# - Include weights with loss function: ```BCEWithLogitsLossFlat(pos_weight=dls.train.cws)```
#   - Weights must be in tensor form and placed on GPU (done by default is used dls.train.cws)
#   - Strict weighting by False/True ratio may yield sub-optimal results.  Consider reduced weights: ```BCEWithLogitsLossFlat(pos_weight=dls.train.cws.sqrt())```
