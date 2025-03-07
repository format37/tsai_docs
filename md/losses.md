## On this page

  * HuberLoss
  * LogCoshLoss
  * MaskedLossWrapper
  * CenterPlusLoss
  * CenterLoss
  * FocalLoss
  * TweedieLoss



  * __Report an issue



  1. Training
  2. Losses



# Losses

> Losses not available in fastai or Pytorch.

* * *

source

### HuberLoss

> 
>      HuberLoss (reduction='mean', delta=1.0)

*Huber loss

Creates a criterion that uses a squared term if the absolute element-wise error falls below delta and a delta-scaled L1 term otherwise. This loss combines advantages of both :class:`L1Loss` and :class:`MSELoss`; the delta-scaled L1 region makes the loss less sensitive to outliers than :class:`MSELoss`, while the L2 region provides smoothness over :class:`L1Loss` near 0. See `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`_ for more information. This loss is equivalent to nn.SmoothL1Loss when delta == 1.*

* * *

source

### LogCoshLoss

> 
>      LogCoshLoss (reduction='mean', delta=1.0)

*Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes::
    
    
    import torch.nn as nn
    import torch.nn.functional as F
    
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)
    
        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will have their parameters converted too when you call :meth:`to`, etc.

.. note:: As per the example above, an `__init__()` call to the parent class must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or evaluation mode. :vartype training: bool*
    
    
    inp = torch.rand(8, 3, 10)
    targ = torch.randn(8, 3, 10)
    test_close(HuberLoss(delta=1)(inp, targ), nn.SmoothL1Loss()(inp, targ))
    LogCoshLoss()(inp, targ)__
    
    
    tensor(0.4588)

* * *

source

### MaskedLossWrapper

> 
>      MaskedLossWrapper (crit)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    inp = torch.rand(8, 3, 10)
    targ = torch.randn(8, 3, 10)
    targ[targ >.8] = np.nan
    nn.L1Loss()(inp, targ), MaskedLossWrapper(nn.L1Loss())(inp, targ)__
    
    
    (tensor(nan), tensor(1.0520))

* * *

source

### CenterPlusLoss

> 
>      CenterPlusLoss (loss, c_out, λ=0.01, logits_dim=None)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### CenterLoss

> 
>      CenterLoss (c_out, logits_dim=None)

*Code in Pytorch has been slightly modified from: https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/center_loss.py Based on paper: Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

Args: c_out (int): number of classes. logits_dim (int): dim 1 of the logits. By default same as c_out (for one hot encoded logits)*
    
    
    c_in = 10
    x = torch.rand(64, c_in).to(device=default_device())
    x = F.softmax(x, dim=1)
    label = x.max(dim=1).indices
    CenterLoss(c_in).to(x.device)(x, label), CenterPlusLoss(LabelSmoothingCrossEntropyFlat(), c_in).to(x.device)(x, label)__
    
    
    (tensor(9.2481, grad_fn=<DivBackward0>),
     TensorBase(2.3559, grad_fn=<AliasBackward0>))
    
    
    CenterPlusLoss(LabelSmoothingCrossEntropyFlat(), c_in)__
    
    
    CenterPlusLoss(loss=FlattenedLoss of LabelSmoothingCrossEntropy(), c_out=10, λ=0.01)

* * *

source

### FocalLoss

> 
>      FocalLoss (alpha:Optional[torch.Tensor]=None, gamma:float=2.0,
>                 reduction:str='mean')

_Weighted, multiclass focal loss_
    
    
    inputs = torch.normal(0, 2, (16, 2)).to(device=default_device())
    targets = torch.randint(0, 2, (16,)).to(device=default_device())
    FocalLoss()(inputs, targets)__
    
    
    tensor(0.9829)

* * *

source

### TweedieLoss

> 
>      TweedieLoss (p=1.5, eps=1e-08)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    c_in = 10
    output = torch.rand(64).to(device=default_device())
    target = torch.rand(64).to(device=default_device())
    TweedieLoss().to(output.device)(output, target)__
    
    
    tensor(3.0539)

  * __Report an issue


