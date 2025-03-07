## On this page

  * SeriesDecomposition
  * MovingAverage
  * Flatten_Head
  * PatchTST
  * Test conversion to Torchscript
  * Test conversion to onnx



  * __Report an issue



  1. Models
  2. Transformers
  3. PatchTST



# PatchTST

This is an unofficial PyTorch implementation of PatchTST created by Ignacio Oguiza (oguiza@timeseriesAI.co) based on:

In this notebook, we are going to use a new state-of-the-art model called PatchTST (Nie et al, 2022) to create a long-term time series forecast.

Here are some paper details:

  * Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2022). **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.** arXiv preprint arXiv:2211.14730.
  * Official implementation:: https://github.com/yuqinie98/PatchTST


    
    
    @article{Yuqietal-2022-PatchTST,
      title={A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
      author={Yuqi Nie and 
              Nam H. Nguyen and 
              Phanwadee Sinthong and 
              Jayant Kalagnanam},
      journal={arXiv preprint arXiv:2211.14730},
      year={2022}
    }__

PatchTST has shown some impressive results across some of the most widely used long-term datasets for benchmarking:

image.png

* * *

source

### SeriesDecomposition

> 
>      SeriesDecomposition (kernel_size:int)

_Series decomposition block_

| **Type** | **Details**  
---|---|---  
kernel_size | int | the size of the window  
  
* * *

source

### MovingAverage

> 
>      MovingAverage (kernel_size:int)

_Moving average block to highlight the trend of time series_

| **Type** | **Details**  
---|---|---  
kernel_size | int | the size of the window  
  
* * *

source

### Flatten_Head

> 
>      Flatten_Head (individual, n_vars, nf, pred_dim)

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

* * *

source

### PatchTST

> 
>      PatchTST (c_in, c_out, seq_len, pred_dim=None, n_layers=2, n_heads=8,
>                d_model=512, d_ff=2048, dropout=0.05, attn_dropout=0.0,
>                patch_len=16, stride=8, padding_patch=True, revin=True,
>                affine=False, individual=False, subtract_last=False,
>                decomposition=False, kernel_size=25, activation='gelu',
>                norm='BatchNorm', pre_norm=False, res_attention=True,
>                store_attn=False)

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

ivar training: Boolean represents whether this module is in training or evaluation mode. :vartype training: bool* | **Type** | **Default** | **Details**  
---|---|---|---  
c_in |  |  | number of input channels  
c_out |  |  | used for compatibility  
seq_len |  |  | input sequence length  
pred_dim | NoneType | None | prediction sequence length  
n_layers | int | 2 | number of encoder layers  
n_heads | int | 8 | number of heads  
d_model | int | 512 | dimension of model  
d_ff | int | 2048 | dimension of fully connected network (fcn)  
dropout | float | 0.05 | dropout applied to all linear layers in the encoder  
attn_dropout | float | 0.0 | dropout applied to the attention scores  
patch_len | int | 16 | patch_len  
stride | int | 8 | stride  
padding_patch | bool | True | flag to indicate if padded is added if necessary  
revin | bool | True | RevIN  
affine | bool | False | RevIN affine  
individual | bool | False | individual head  
subtract_last | bool | False | subtract_last  
decomposition | bool | False | apply decomposition  
kernel_size | int | 25 | decomposition kernel size  
activation | str | gelu | activation function of intermediate layer, relu or gelu.  
norm | str | BatchNorm | type of normalization layer used in the encoder  
pre_norm | bool | False | flag to indicate if normalization is applied as the first step in the sublayers  
res_attention | bool | True | flag to indicate if Residual MultiheadAttention should be used  
store_attn | bool | False | can be used to visualize attention weights  
      
    
    from fastcore.test import test_eq
    from tsai.models.utils import count_parameters
    
    bs = 32
    c_in = 9  # aka channels, features, variables, dimensions
    c_out = 1
    seq_len = 60
    pred_dim = 20
    
    xb = torch.randn(bs, c_in, seq_len)
    
    arch_config=dict(
            n_layers=3,  # number of encoder layers
            n_heads=16,  # number of heads
            d_model=128,  # dimension of model
            d_ff=256,  # dimension of fully connected network (fcn)
            attn_dropout=0.,
            dropout=0.2,  # dropout applied to all linear layers in the encoder
            patch_len=16,  # patch_len
            stride=8,  # stride
        )
    
    model = PatchTST(c_in, c_out, seq_len, pred_dim, **arch_config)
    test_eq(model.to(xb.device)(xb).shape, [bs, c_in, pred_dim])
    print(f'model parameters: {count_parameters(model)}')__
    
    
    model parameters: 418470

### Test conversion to Torchscript
    
    
    import gc
    import os
    import torch
    import torch.nn as nn
    from fastcore.test import test_eq, test_close
    
    
    bs = 1
    new_bs = 8
    c_in = 3
    c_out = 1
    seq_len = 96
    pred_dim = 20
    
    # module
    model = PatchTST(c_in, c_out, seq_len, pred_dim)
    model = model.eval()
    
    # input data
    inp = torch.rand(bs, c_in, seq_len)
    new_inp = torch.rand(new_bs, c_in, seq_len)
    
    # original
    try:
        output = model(inp)
        new_output = model(new_inp)
        print(f'{"original":10}: ok')
    except:
        print(f'{"original":10}: failed')
    
    # tracing
    try:
        traced_model = torch.jit.trace(model, inp)
        file_path = f"_test_traced_model.pt"
        torch.jit.save(traced_model, file_path)
        traced_model = torch.jit.load(file_path)
        test_eq(output, traced_model(inp))
        test_eq(new_output, traced_model(new_inp))
        os.remove(file_path)
        del traced_model
        gc.collect()
        print(f'{"tracing":10}: ok')
    except:
        print(f'{"tracing":10}: failed')
    
    # scripting
    try:
        scripted_model = torch.jit.script(model)
        file_path = f"_test_scripted_model.pt"
        torch.jit.save(scripted_model, file_path)
        scripted_model = torch.jit.load(file_path)
        test_eq(output, scripted_model(inp))
        test_eq(new_output, scripted_model(new_inp))
        os.remove(file_path)
        del scripted_model
        gc.collect()
        print(f'{"scripting":10}: ok')
    except:
        print(f'{"scripting":10}: failed')__
    
    
    original  : ok
    tracing   : ok
    scripting : failed

### Test conversion to onnx
    
    
    try:
        import onnx
        import onnxruntime as ort
        
        try:
            file_path = "_model_cpu.onnx"
            torch.onnx.export(model.cpu(),               # model being run
                            inp,                       # model input (or a tuple for multiple inputs)
                            file_path,                 # where to save the model (can be a file or file-like object)
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={
                                'input'  : {0 : 'batch_size'}, 
                                'output' : {0 : 'batch_size'}} # variable length axes
                            )
    
    
            # Load the model and check it's ok
            onnx_model = onnx.load(file_path)
            onnx.checker.check_model(onnx_model)
            del onnx_model
            gc.collect()
    
            # New session
            ort_sess = ort.InferenceSession(file_path)
            output_onnx = ort_sess.run(None, {'input': inp.numpy()})[0]
            test_close(output.detach().numpy(), output_onnx)
            new_output_onnx = ort_sess.run(None, {'input': new_inp.numpy()})[0]
            test_close(new_output.detach().numpy(), new_output_onnx)
            os.remove(file_path)
            print(f'{"onnx":10}: ok')
        except:
            print(f'{"onnx":10}: failed')
    
    except ImportError:
        print('onnx and onnxruntime are not installed. Please install them to run this test')__
    
    
    onnx and onnxruntime are not installed. Please install them to run this test

  * __Report an issue


