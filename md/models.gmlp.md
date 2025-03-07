## On this page

  * gMLP



  * __Report an issue



  1. Models
  2. MLPs
  3. gMLP



# gMLP

This is an unofficial PyTorch implementation based on:

  * Liu, H., Dai, Z., So, D. R., & Le, Q. V. (2021). **Pay Attention to MLPs**. arXiv preprint arXiv:2105.08050.

  * Cholakov, R., & Kolev, T. (2022). **The GatedTabTransformer. An enhanced deep learning architecture for tabular modeling**. arXiv preprint arXiv:2201.00199.




* * *

source

### gMLP

> 
>      gMLP (c_in, c_out, seq_len, patch_size=1, d_model=256, d_ffn=512,
>            depth=6)

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
    
    
    bs = 16
    c_in = 3
    c_out = 2
    seq_len = 64
    patch_size = 4
    xb = torch.rand(bs, c_in, seq_len)
    model = gMLP(c_in, c_out, seq_len, patch_size=patch_size)
    test_eq(model(xb).shape, (bs, c_out))__

  * __Report an issue


