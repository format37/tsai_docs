## On this page

  * NoisyStudent



  * __Report an issue



  1. Training
  2. Callbacks
  3. Noisy student



# Noisy student

Callback to apply noisy student self-training (a semi-supervised learning approach) based on:

Xie, Q., Luong, M. T., Hovy, E., & Le, Q. V. (2020). Self-training with noisy student improves imagenet classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10687-10698).

* * *

source

### NoisyStudent

> 
>      NoisyStudent (dl2:fastai.data.load.DataLoader, bs:Optional[int]=None,
>                    l2pl_ratio:int=1, batch_tfms:Optional[list]=None,
>                    do_setup:bool=True, pseudolabel_sample_weight:float=1.0,
>                    verbose=False)

*A callback to implement the Noisy Student approach. In the original paper this was used in combination with noise: - stochastic depth: .8 - RandAugment: N=2, M=27 - dropout: .5

Steps: 1. Build the dl you will use as a teacher 2. Create dl2 with the pseudolabels (either soft or hard preds) 3. Pass any required batch_tfms to the callback*
    
    
    from tsai.data.all import *
    from tsai.models.all import *
    from tsai.tslearner import *__
    
    
    dsid = 'NATOPS'
    X, y, splits = get_UCR_data(dsid, return_split=False)
    X = X.astype(np.float32)__
    
    
    pseudolabeled_data = X
    soft_preds = True
    
    pseudolabels = ToNumpyCategory()(y) if soft_preds else OneHot()(y)
    dsets2 = TSDatasets(pseudolabeled_data, pseudolabels)
    dl2 = TSDataLoader(dsets2, num_workers=0)
    noisy_student_cb = NoisyStudent(dl2, bs=256, l2pl_ratio=2, verbose=True)
    tfms = [None, TSClassification]
    learn = TSClassifier(X, y, splits=splits, tfms=tfms, batch_tfms=[TSStandardize(), TSRandomSize(.5)], cbs=noisy_student_cb)
    learn.fit_one_cycle(1)__
    
    
    labels / pseudolabels per training batch              : 171 / 85
    relative labeled/ pseudolabel sample weight in dataset: 4.0

epoch | train_loss | valid_loss | accuracy | time  
---|---|---|---|---  
0 | 1.782144 | 1.758471 | 0.250000 | 00:00  
      
    
    X: torch.Size([171, 24, 51])  X2: torch.Size([85, 24, 51])  X_comb: torch.Size([256, 24, 41])
    y: torch.Size([171])  y2: torch.Size([85])  y_comb: torch.Size([256])
    
    
    pseudolabeled_data = X
    soft_preds = False
    
    pseudolabels = ToNumpyCategory()(y) if soft_preds else OneHot()(y)
    pseudolabels = pseudolabels.astype(np.float32)
    dsets2 = TSDatasets(pseudolabeled_data, pseudolabels)
    dl2 = TSDataLoader(dsets2, num_workers=0)
    noisy_student_cb = NoisyStudent(dl2, bs=256, l2pl_ratio=2, verbose=True)
    tfms = [None, TSClassification]
    learn = TSClassifier(X, y, splits=splits, tfms=tfms, batch_tfms=[TSStandardize(), TSRandomSize(.5)], cbs=noisy_student_cb)
    learn.fit_one_cycle(1)__
    
    
    labels / pseudolabels per training batch              : 171 / 85
    relative labeled/ pseudolabel sample weight in dataset: 4.0

epoch | train_loss | valid_loss | accuracy | time  
---|---|---|---|---  
0 | 1.898401 | 1.841182 | 0.155556 | 00:00  
      
    
    X: torch.Size([171, 24, 51])  X2: torch.Size([85, 24, 51])  X_comb: torch.Size([256, 24, 51])
    y: torch.Size([171, 6])  y2: torch.Size([85, 6])  y_comb: torch.Size([256, 6])

  * __Report an issue


