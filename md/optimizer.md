## On this page

  * wrap_optimizer



  * __Report an issue



  1. Training
  2. Optimizers



# Optimizers

> This contains a set of optimizers.

* * *

source

### wrap_optimizer

> 
>      wrap_optimizer (opt, **kwargs)

You can natively use any of the optimizers included in the fastai library. You just need to pass it to the learner as the opt_func.

In addition, you will be able to use any of the optimizers from:

  * Pytorch
  * torch_optimizer (https://github.com/jettify/pytorch-optimizer). In this case, you will need to install `torch-optimizer` first)



Examples of use:
    
    
    adamw = wrap_optimizer(torch.optim.AdamW)__
    
    
     import torch_optimizer as optim
    adabelief = wrap_optimizer(optim.AdaBelief)__

If you want to use any these last 2, you can use the wrap_optimizer function. Here are a few examples:

  * __Report an issue


