## On this page

  * InputWrapper
  * ResidualWrapper
  * RecursiveWrapper



  * __Report an issue



  1. Models
  2. Miscellaneous
  3. Miscellaneous



# Miscellaneous

> This contains a set of experiments.

* * *

source

### InputWrapper

> 
>      InputWrapper (arch, c_in, c_out, seq_len, new_c_in=None,
>                    new_seq_len=None, **kwargs)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    from tsai.models.TST import *__
    
    
    xb = torch.randn(16, 1, 1000)
    model = InputWrapper(TST, 1, 4, 1000, 10, 224)
    test_eq(model.to(xb.device)(xb).shape, (16,4))__

* * *

source

### ResidualWrapper

> 
>      ResidualWrapper (model)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### RecursiveWrapper

> 
>      RecursiveWrapper (model, n_steps, anchored=False)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    xb = torch.randn(16, 1, 20)
    model = RecursiveWrapper(TST(1, 1, 20), 5)
    test_eq(model.to(xb.device)(xb).shape, (16, 5))__

  * __Report an issue


