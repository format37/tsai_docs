## On this page

  * test_module_to_torchscript
  * init_lin_zero
  * SwishBeta
  * SmeLU
  * Chomp1d
  * SameConv1d
  * Pad1d
  * same_padding1d
  * Conv2d
  * Conv2dSame
  * Pad2d
  * same_padding2d
  * CausalConv1d
  * Conv1d
  * SeparableConv1d
  * AddCoords1d
  * ConvBlock
  * ResBlock1dPlus
  * SEModule1d
  * Norm
  * LinLnDrop
  * LambdaPlus
  * ReZero
  * Clip
  * Clamp
  * SoftMax
  * LastStep
  * Max
  * Reshape
  * View
  * Transpose
  * Permute
  * Unfold
  * Concat
  * Add
  * Unsqueeze
  * Squeeze
  * DropPath
  * Sharpen
  * Sequential
  * TimeDistributed
  * get_calibrator
  * Matrix_Scale
  * Vector_Scale
  * Temp_Scale
  * LogitAdjustmentLayer
  * MaxPPVPool1d
  * PPAuc
  * PPV
  * AdaptiveWeightedAvgPool1d
  * GAWP1d
  * GACP1d
  * GAP1d
  * gwa_pool_head
  * GlobalWeightedAveragePool1d
  * attentional_pool_head
  * GAttP1d
  * AttentionalPool1d
  * PoolingLayer
  * ReGLU
  * GEGLU
  * get_act_fn
  * RevIN
  * RevIN
  * create_pool_head
  * max_pool_head
  * create_pool_plus_head
  * create_conv_head
  * create_mlp_head
  * create_fc_head
  * create_rnn_head
  * imputation_head
  * create_conv_lin_nd_head
  * lin_nd_head
  * rocket_nd_head
  * xresnet1d_nd_head
  * create_conv_3d_head
  * universal_pool_head
  * SqueezeExciteBlock
  * GaussianNoise
  * TokenLayer
  * PositionwiseFeedForward
  * ScaledDotProductAttention
  * MultiheadAttention
  * MultiConv1d
  * LSTMOutput
  * emb_sz_rule
  * TSEmbedding
  * MultiEmbedding



  * __Report an issue



  1. Models
  2. Layers



# Layers

> Helper functions used to build PyTorch timeseries models.

* * *

source

### test_module_to_torchscript

> 
>      test_module_to_torchscript (m:torch.nn.modules.module.Module,
>                                  inputs:torch.Tensor, trace:bool=True,
>                                  script:bool=True, serialize:bool=True,
>                                  verbose:bool=True)

_Tests if a PyTorch module can be correctly traced or scripted and serialized_

| **Type** | **Default** | **Details**  
---|---|---|---  
m | Module |  | The PyTorch module to be tested.  
inputs | Tensor |  | A tensor or tuple of tensors representing the inputs to the model.  
trace | bool | True | If `True`, attempts to trace the model. Defaults to `True`.  
script | bool | True | If `True`, attempts to script the model. Defaults to `True`.  
serialize | bool | True | If `True`, saves and loads the traced/scripted module to ensure it can be serialized. Defaults to `True`.  
verbose | bool | True | If `True`, prints detailed information about the tracing and scripting process. Defaults to `True`.  
      
    
    m = nn.Linear(10, 2)
    inp = torch.randn(3, 10)
    test_module_to_torchscript(m, inp, trace=True, script=True, serialize=True, verbose=True)__
    
    
    output.shape: torch.Size([3, 2])
    Tracing...
    ...Linear has been successfully traced 😃
    
    
    
    True

* * *

source

### init_lin_zero

> 
>      init_lin_zero (m)

* * *

source

### SwishBeta

> 
>      SwishBeta ()

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### SmeLU

> 
>      SmeLU (beta:float=2.0)

_Smooth ReLU activation function based on https://arxiv.org/pdf/2202.06499.pdf_

| **Type** | **Default** | **Details**  
---|---|---|---  
beta | float | 2.0 | Beta value  
**Returns** | **None** |  |   
  
* * *

source

### Chomp1d

> 
>      Chomp1d (chomp_size)

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

### SameConv1d

> 
>      SameConv1d (ni, nf, ks=3, stride=1, dilation=1, **kwargs)

_Conv1d with padding=‘same’_

* * *

source

### Pad1d

> 
>      Pad1d (padding, value=0.0)

*Pads the input tensor boundaries with a constant value.

For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

Args: padding (int, tuple): the size of the padding. If is `int`, uses the same padding in both boundaries. If a 2-`tuple`, uses (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`)

Shape: - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`. - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where
    
    
      :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

Examples::
    
    
    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> m = nn.ConstantPad1d(2, 3.5)
    >>> input = torch.randn(1, 2, 4)
    >>> input
    tensor([[[-1.0491, -0.7152, -0.0749,  0.8530],
             [-1.3287,  1.8966,  0.1466, -0.2771]]])
    >>> m(input)
    tensor([[[ 3.5000,  3.5000, -1.0491, -0.7152, -0.0749,  0.8530,  3.5000,
               3.5000],
             [ 3.5000,  3.5000, -1.3287,  1.8966,  0.1466, -0.2771,  3.5000,
               3.5000]]])
    >>> m = nn.ConstantPad1d(2, 3.5)
    >>> input = torch.randn(1, 2, 3)
    >>> input
    tensor([[[ 1.6616,  1.4523, -1.1255],
             [-3.6372,  0.1182, -1.8652]]])
    >>> m(input)
    tensor([[[ 3.5000,  3.5000,  1.6616,  1.4523, -1.1255,  3.5000,  3.5000],
             [ 3.5000,  3.5000, -3.6372,  0.1182, -1.8652,  3.5000,  3.5000]]])
    >>> # using different paddings for different sides
    >>> m = nn.ConstantPad1d((3, 1), 3.5)
    >>> m(input)
    tensor([[[ 3.5000,  3.5000,  3.5000,  1.6616,  1.4523, -1.1255,  3.5000],
             [ 3.5000,  3.5000,  3.5000, -3.6372,  0.1182, -1.8652,  3.5000]]])*

* * *

source

### same_padding1d

> 
>      same_padding1d (seq_len, ks, stride=1, dilation=1)

_Same padding formula as used in Tensorflow_

* * *

source

### Conv2d

> 
>      Conv2d (ni, nf, kernel_size=None, ks=None, stride=1, padding='same',
>              dilation=1, init='auto', bias_std=0.01, **kwargs)

_conv1d layer with padding=‘same’, ‘valid’, or any integer (defaults to ‘same’)_

* * *

source

### Conv2dSame

> 
>      Conv2dSame (ni, nf, ks=(3, 3), stride=(1, 1), dilation=(1, 1), **kwargs)

_Conv2d with padding=‘same’_

* * *

source

### Pad2d

> 
>      Pad2d (padding, value=0.0)

*Pads the input tensor boundaries with a constant value.

For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

Args: padding (int, tuple): the size of the padding. If is `int`, uses the same padding in all boundaries. If a 4-`tuple`, uses (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)

Shape: - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`. - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where
    
    
      :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`
    
      :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

Examples::
    
    
    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> m = nn.ConstantPad2d(2, 3.5)
    >>> input = torch.randn(1, 2, 2)
    >>> input
    tensor([[[ 1.6585,  0.4320],
             [-0.8701, -0.4649]]])
    >>> m(input)
    tensor([[[ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
             [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
             [ 3.5000,  3.5000,  1.6585,  0.4320,  3.5000,  3.5000],
             [ 3.5000,  3.5000, -0.8701, -0.4649,  3.5000,  3.5000],
             [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
             [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000]]])
    >>> # using different paddings for different sides
    >>> m = nn.ConstantPad2d((3, 0, 2, 1), 3.5)
    >>> m(input)
    tensor([[[ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
             [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
             [ 3.5000,  3.5000,  3.5000,  1.6585,  0.4320],
             [ 3.5000,  3.5000,  3.5000, -0.8701, -0.4649],
             [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000]]])*

* * *

source

### same_padding2d

> 
>      same_padding2d (H, W, ks, stride=(1, 1), dilation=(1, 1))

_Same padding formula as used in Tensorflow_
    
    
    bs = 2
    c_in = 3
    c_out = 5
    h = 16
    w = 20
    t = torch.rand(bs, c_in, h, w)
    test_eq(Conv2dSame(c_in, c_out, ks=3, stride=1, dilation=1, bias=False)(t).shape, (bs, c_out, h, w))
    test_eq(Conv2dSame(c_in, c_out, ks=(3, 1), stride=1, dilation=1, bias=False)(t).shape, (bs, c_out, h, w))
    test_eq(Conv2dSame(c_in, c_out, ks=3, stride=(1, 1), dilation=(2, 2), bias=False)(t).shape, (bs, c_out, h, w))
    test_eq(Conv2dSame(c_in, c_out, ks=3, stride=(2, 2), dilation=(1, 1), bias=False)(t).shape, (bs, c_out, h//2, w//2))
    test_eq(Conv2dSame(c_in, c_out, ks=3, stride=(2, 2), dilation=(2, 2), bias=False)(t).shape, (bs, c_out, h//2, w//2))
    test_eq(Conv2d(c_in, c_out, ks=3, padding='same', stride=1, dilation=1, bias=False)(t).shape, (bs, c_out, h, w))__

* * *

source

### CausalConv1d

> 
>      CausalConv1d (ni, nf, ks, stride=1, dilation=1, groups=1, bias=True)

*Applies a 1D convolution over an input signal composed of several input planes.

In the simplest case, the output value of the layer with input size :math:`(N, C_{\text{in}}, L)` and output :math:`(N, C_{\text{out}}, L_{\text{out}})` can be precisely described as:

.. math:: (N_i, C_{_j}) = (C_{_j}) +_{k = 0}^{C_{in} - 1} (C_{_j}, k) (N_i, k)

where :math:`\star` is the valid `cross-correlation`_ operator, :math:`N` is a batch size, :math:`C` denotes a number of channels, :math:`L` is a length of signal sequence.

This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

  * :attr:`stride` controls the stride for the cross-correlation, a single number or a one-element tuple.

  * :attr:`padding` controls the amount of padding applied to the input. It can be either a string {‘valid’, ‘same’} or a tuple of ints giving the amount of implicit padding applied on both sides.

  * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

  * :attr:`groups` controls the connections between inputs and outputs. :attr:`in_channels` and :attr:`out_channels` must both be divisible by :attr:`groups`. For example,

    * At groups=1, all inputs are convolved to all outputs.
    * At groups=2, the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels and producing half the output channels, and both subsequently concatenated.
    * At groups= :attr:`in_channels`, each input channel is convolved with its own set of filters (of size :math:`\frac{\text{out\_channels}}{\text{in\_channels}}`).



Note: When `groups == in_channels` and `out_channels == K * in_channels`, where `K` is a positive integer, this operation is also known as a “depthwise convolution”.
    
    
    In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
    a depthwise convolution with a depthwise multiplier `K` can be performed with the arguments
    :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})`.

Note: In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting `torch.backends.cudnn.deterministic = True`. See :doc:`/notes/randomness` for more information.

Note: `padding='valid'` is the same as no padding. `padding='same'` pads the input so the output has the shape as the input. However, this mode doesn’t support any stride values other than 1.

Note: This module supports complex data types i.e. `complex32, complex64, complex128`.

Args: in_channels (int): Number of channels in the input image out_channels (int): Number of channels produced by the convolution kernel_size (int or tuple): Size of the convolving kernel stride (int or tuple, optional): Stride of the convolution. Default: 1 padding (int, tuple or str, optional): Padding added to both sides of the input. Default: 0 dilation (int or tuple, optional): Spacing between kernel elements. Default: 1 groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1 bias (bool, optional): If `True`, adds a learnable bias to the output. Default: `True` padding_mode (str, optional): `'zeros'`, `'reflect'`, `'replicate'` or `'circular'`. Default: `'zeros'`

Shape: - Input: :math:`(N, C_{in}, L_{in})` or :math:`(C_{in}, L_{in})` \- Output: :math:`(N, C_{out}, L_{out})` or :math:`(C_{out}, L_{out})`, where
    
    
      .. math::
          L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                    \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

Attributes: weight (Tensor): the learnable weights of the module of shape :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size})`. The values of these weights are sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}` bias (Tensor): the learnable bias of the module of shape (out_channels). If :attr:`bias` is `True`, then the values of these weights are sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`

Examples::
    
    
    >>> m = nn.Conv1d(16, 33, 3, stride=2)
    >>> input = torch.randn(20, 16, 50)
    >>> output = m(input)

.. _cross-correlation: https://en.wikipedia.org/wiki/Cross-correlation

.. _link: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md*

* * *

source

### Conv1d

> 
>      Conv1d (ni, nf, kernel_size=None, ks=None, stride=1, padding='same',
>              dilation=1, init='auto', bias_std=0.01, **kwargs)

_conv1d layer with padding=‘same’, ‘causal’, ‘valid’, or any integer (defaults to ‘same’)_
    
    
    bs = 2
    c_in = 3
    c_out = 5
    seq_len = 512
    t = torch.rand(bs, c_in, seq_len)
    dilation = 1
    test_eq(CausalConv1d(c_in, c_out, ks=3, dilation=dilation)(t).shape, Conv1d(c_in, c_out, ks=3, padding="same", dilation=dilation)(t).shape)
    dilation = 2
    test_eq(CausalConv1d(c_in, c_out, ks=3, dilation=dilation)(t).shape, Conv1d(c_in, c_out, ks=3, padding="same", dilation=dilation)(t).shape)__
    
    
    bs = 2
    ni = 3
    nf = 5
    seq_len = 6
    ks = 3
    t = torch.rand(bs, c_in, seq_len)
    test_eq(Conv1d(ni, nf, ks, padding=0)(t).shape, (bs, c_out, seq_len - (2 * (ks//2))))
    test_eq(Conv1d(ni, nf, ks, padding='valid')(t).shape, (bs, c_out, seq_len - (2 * (ks//2))))
    test_eq(Conv1d(ni, nf, ks, padding='same')(t).shape, (bs, c_out, seq_len))
    test_eq(Conv1d(ni, nf, ks, padding='causal')(t).shape, (bs, c_out, seq_len))
    test_error('use kernel_size or ks but not both simultaneously', Conv1d, ni, nf, kernel_size=3, ks=3)
    test_error('you need to pass a ks', Conv1d, ni, nf)__
    
    
    conv = Conv1d(ni, nf, ks, padding='same')
    init_linear(conv, None, init='auto', bias_std=.01)
    conv __
    
    
    Conv1d(3, 5, kernel_size=(3,), stride=(1,), padding=(1,))
    
    
    conv = Conv1d(ni, nf, ks, padding='causal')
    init_linear(conv, None, init='auto', bias_std=.01)
    conv __
    
    
    CausalConv1d(3, 5, kernel_size=(3,), stride=(1,))
    
    
    conv = Conv1d(ni, nf, ks, padding='valid')
    init_linear(conv, None, init='auto', bias_std=.01)
    weight_norm(conv)
    conv __
    
    
    Conv1d(3, 5, kernel_size=(3,), stride=(1,))
    
    
    conv = Conv1d(ni, nf, ks, padding=0)
    init_linear(conv, None, init='auto', bias_std=.01)
    weight_norm(conv)
    conv __
    
    
    Conv1d(3, 5, kernel_size=(3,), stride=(1,))

* * *

source

### SeparableConv1d

> 
>      SeparableConv1d (ni, nf, ks, stride=1, padding='same', dilation=1,
>                       bias=True, bias_std=0.01)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    bs = 64
    c_in = 6
    c_out = 5
    seq_len = 512
    t = torch.rand(bs, c_in, seq_len)
    test_eq(SeparableConv1d(c_in, c_out, 3)(t).shape, (bs, c_out, seq_len))__

* * *

source

### AddCoords1d

> 
>      AddCoords1d ()

_Add coordinates to ease position identification without modifying mean and std_
    
    
    bs = 2
    c_in = 3
    c_out = 5
    seq_len = 50
    
    t = torch.rand(bs, c_in, seq_len)
    t = (t - t.mean()) / t.std()
    test_eq(AddCoords1d()(t).shape, (bs, c_in + 1, seq_len))
    new_t = AddCoords1d()(t)
    test_close(new_t.mean(),0, 1e-2)
    test_close(new_t.std(), 1, 1e-2)__

* * *

source

### ConvBlock

> 
>      ConvBlock (ni, nf, kernel_size=None, ks=3, stride=1, padding='same',
>                 bias=None, bias_std=0.01, norm='Batch', zero_norm=False,
>                 bn_1st=True, act=<class 'torch.nn.modules.activation.ReLU'>,
>                 act_kwargs={}, init='auto', dropout=0.0, xtra=None,
>                 coord=False, separable=False, **kwargs)

_Create a sequence of conv1d (`ni` to `nf`), activation (if `act_cls`) and `norm_type` layers._

* * *

source

### ResBlock1dPlus

> 
>      ResBlock1dPlus (expansion, ni, nf, coord=False, stride=1, groups=1,
>                      reduction=None, nh1=None, nh2=None, dw=False, g2=1,
>                      sa=False, sym=False, norm='Batch', zero_norm=True,
>                      act_cls=<class 'torch.nn.modules.activation.ReLU'>, ks=3,
>                      pool=<function AvgPool>, pool_first=True, **kwargs)

_Resnet block from`ni` to `nh` with `stride`_

* * *

source

### SEModule1d

> 
>      SEModule1d (ni, reduction=16, act=<class
>                  'torch.nn.modules.activation.ReLU'>, act_kwargs={})

_Squeeze and excitation module for 1d_
    
    
    t = torch.rand(8, 32, 12)
    test_eq(SEModule1d(t.shape[1], 16, act=nn.ReLU, act_kwargs={})(t).shape, t.shape)__

* * *

source

### Norm

> 
>      Norm (nf, ndim=1, norm='Batch', zero_norm=False, init=True, **kwargs)

_Norm layer with`nf` features and `ndim` with auto init._
    
    
    bs = 2
    ni = 3
    nf = 5
    sl = 4
    ks = 5
    
    t = torch.rand(bs, ni, sl)
    test_eq(ConvBlock(ni, nf, ks)(t).shape, (bs, nf, sl))
    test_eq(ConvBlock(ni, nf, ks, padding='causal')(t).shape, (bs, nf, sl))
    test_eq(ConvBlock(ni, nf, ks, coord=True)(t).shape, (bs, nf, sl))__
    
    
    test_eq(BN1d(ni)(t).shape, (bs, ni, sl))
    test_eq(BN1d(ni).weight.data.mean().item(), 1.)
    test_eq(BN1d(ni, zero_norm=True).weight.data.mean().item(), 0.)__
    
    
    test_eq(ConvBlock(ni, nf, ks, norm='batch', zero_norm=True)[1].weight.data.unique().item(), 0)
    test_ne(ConvBlock(ni, nf, ks, norm='batch', zero_norm=False)[1].weight.data.unique().item(), 0)
    test_eq(ConvBlock(ni, nf, ks, bias=False)[0].bias, None)
    ConvBlock(ni, nf, ks, act=Swish, coord=True)__
    
    
    ConvBlock(
      (0): AddCoords1d()
      (1): Conv1d(4, 5, kernel_size=(5,), stride=(1,), padding=(2,), bias=False)
      (2): BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): SiLU()
    )

* * *

source

### LinLnDrop

> 
>      LinLnDrop (n_in, n_out, ln=True, p=0.0, act=None, lin_first=False)

_Module grouping`LayerNorm1d`, `Dropout` and `Linear` layers_
    
    
    LinLnDrop(2, 3, p=.5)__
    
    
    LinLnDrop(
      (0): LayerNorm((2,), eps=1e-05, elementwise_affine=True)
      (1): Dropout(p=0.5, inplace=False)
      (2): Linear(in_features=2, out_features=3, bias=False)
    )

* * *

source

### LambdaPlus

> 
>      LambdaPlus (func, *args, **kwargs)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### ReZero

> 
>      ReZero (module)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### Clip

> 
>      Clip (min=None, max=None)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### Clamp

> 
>      Clamp (min=None, max=None)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### SoftMax

> 
>      SoftMax (dim=-1)

_SoftMax layer_

* * *

source

### LastStep

> 
>      LastStep ()

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### Max

> 
>      Max (dim=None, keepdim=False)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### Reshape

> 
>      Reshape (*shape)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### View

> 
>      View (*shape)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### Transpose

> 
>      Transpose (*dims, contiguous=True)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### Permute

> 
>      Permute (*dims)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### Unfold

> 
>      Unfold (dim, size, step=1)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### Concat

> 
>      Concat (dim=1)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### Add

> 
>      Add ()

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### Unsqueeze

> 
>      Unsqueeze (dim=-1)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### Squeeze

> 
>      Squeeze (dim=-1)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    bs = 2
    nf = 5
    sl = 4
    
    t = torch.rand(bs, nf, sl)
    test_eq(Permute(0,2,1)(t).shape, (bs, sl, nf))
    test_eq(Max(1)(t).shape, (bs, sl))
    test_eq(Transpose(1,2)(t).shape, (bs, sl, nf))
    test_eq(Transpose(1,2, contiguous=True)(t).shape, (bs, sl, nf))
    test_eq(View(-1, 2, 10)(t).shape, (bs, 1, 2, 10))
    test_eq(Reshape(-1, 2, 10)(t).shape, (bs, 1, 2, 10))
    test_eq(Reshape()(t).shape, (2, 20))
    test_eq(Reshape(-1)(t).shape, (40,))
    Transpose(1,2), Permute(0,2,1), View(-1, 2, 10), Transpose(1,2, contiguous=True), Reshape(-1, 2, 10), Noop __
    
    
    (Transpose(dims=1, 2).contiguous(),
     Permute(dims=0, 2, 1),
     View(bs, -1, 2, 10),
     Transpose(dims=1, 2).contiguous(),
     Reshape(bs, -1, 2, 10),
     Sequential())

* * *

source

### DropPath

> 
>      DropPath (p=None)

*Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

It’s similar to Dropout but it drops individual connections instead of nodes. Original code in https://github.com/rwightman/pytorch-image-models (timm library)*
    
    
    t = torch.ones(100,2,3)
    test_eq(DropPath(0.)(t), t)
    assert DropPath(0.5)(t).max() >= 1 __

* * *

source

### Sharpen

> 
>      Sharpen (T=0.5)

_This is used to increase confidence in predictions - MixMatch paper_
    
    
    n_samples = 1000
    n_classes = 3
    
    t = (torch.rand(n_samples, n_classes) - .5) * 10
    probas = F.softmax(t, -1)
    sharpened_probas = Sharpen()(probas)
    plt.plot(probas.flatten().sort().values, color='r')
    plt.plot(sharpened_probas.flatten().sort().values, color='b')
    plt.show()
    test_gt(sharpened_probas[n_samples//2:].max(-1).values.sum().item(), probas[n_samples//2:].max(-1).values.sum().item())__

* * *

source

### Sequential

> 
>      Sequential (*args)

_Class that allows you to pass one or multiple inputs_

* * *

source

### TimeDistributed

> 
>      TimeDistributed (module, batch_first=False)

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

### get_calibrator

> 
>      get_calibrator (calibrator=None, n_classes=1, **kwargs)

* * *

source

### Matrix_Scale

> 
>      Matrix_Scale (n_classes=1, dirichlet=False)

_Used to perform Matrix Scaling (dirichlet=False) or Dirichlet calibration (dirichlet=True)_

* * *

source

### Vector_Scale

> 
>      Vector_Scale (n_classes=1, dirichlet=False)

_Used to perform Vector Scaling (dirichlet=False) or Diagonal Dirichlet calibration (dirichlet=True)_

* * *

source

### Temp_Scale

> 
>      Temp_Scale (temp=1.0, dirichlet=False)

_Used to perform Temperature Scaling (dirichlet=False) or Single-parameter Dirichlet calibration (dirichlet=True)_
    
    
    bs = 2
    c_out = 3
    
    t = torch.rand(bs, c_out)
    for calibrator, cal_name in zip(['temp', 'vector', 'matrix'], ['Temp_Scale', 'Vector_Scale', 'Matrix_Scale']):
        cal = get_calibrator(calibrator, n_classes=c_out)
    #     print(calibrator)
    #     print(cal.weight, cal.bias, '\n')
        test_eq(cal(t), t)
        test_eq(cal.__class__.__name__, cal_name)
    for calibrator, cal_name in zip(['dtemp', 'dvector', 'dmatrix'], ['Temp_Scale', 'Vector_Scale', 'Matrix_Scale']):
        cal = get_calibrator(calibrator, n_classes=c_out)
    #     print(calibrator)
    #     print(cal.weight, cal.bias, '\n')
        test_eq(cal(t), F.log_softmax(t, dim=1))
        test_eq(cal.__class__.__name__, cal_name)__
    
    
    bs = 2
    c_out = 3
    
    t = torch.rand(bs, c_out)
    
    test_eq(Temp_Scale()(t).shape, t.shape)
    test_eq(Vector_Scale(c_out)(t).shape, t.shape)
    test_eq(Matrix_Scale(c_out)(t).shape, t.shape)
    test_eq(Temp_Scale(dirichlet=True)(t).shape, t.shape)
    test_eq(Vector_Scale(c_out, dirichlet=True)(t).shape, t.shape)
    test_eq(Matrix_Scale(c_out, dirichlet=True)(t).shape, t.shape)
    
    test_eq(Temp_Scale()(t), t)
    test_eq(Vector_Scale(c_out)(t), t)
    test_eq(Matrix_Scale(c_out)(t), t)__
    
    
    bs = 2
    c_out = 5
    
    t = torch.rand(bs, c_out)
    test_eq(Vector_Scale(c_out)(t), t)
    test_eq(Vector_Scale(c_out).weight.data, torch.ones(c_out))
    test_eq(Vector_Scale(c_out).weight.requires_grad, True)
    test_eq(type(Vector_Scale(c_out).weight), torch.nn.parameter.Parameter)__
    
    
    bs = 2
    c_out = 3
    weight = 2
    bias = 1
    
    t = torch.rand(bs, c_out)
    test_eq(Matrix_Scale(c_out)(t).shape, t.shape)
    test_eq(Matrix_Scale(c_out).weight.requires_grad, True)
    test_eq(type(Matrix_Scale(c_out).weight), torch.nn.parameter.Parameter)__

* * *

source

### LogitAdjustmentLayer

> 
>      LogitAdjustmentLayer (class_priors)

_Logit Adjustment for imbalanced datasets_
    
    
    bs, n_classes = 16, 3
    class_priors = torch.rand(n_classes)
    logits = torch.randn(bs, n_classes) * 2
    test_eq(LogitAdjLayer(class_priors)(logits), logits + class_priors)__

* * *

source

### MaxPPVPool1d

> 
>      MaxPPVPool1d ()

_Drop-in replacement for AdaptiveConcatPool1d - multiplies nf by 2_

* * *

source

### PPAuc

> 
>      PPAuc (dim=-1)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### PPV

> 
>      PPV (dim=-1)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    bs = 2
    nf = 5
    sl = 4
    
    t = torch.rand(bs, nf, sl)
    test_eq(MaxPPVPool1d()(t).shape, (bs, nf*2, 1))
    test_eq(MaxPPVPool1d()(t).shape, AdaptiveConcatPool1d(1)(t).shape)__

* * *

source

### AdaptiveWeightedAvgPool1d

> 
>      AdaptiveWeightedAvgPool1d (n_in, seq_len, mult=2, n_layers=2, ln=False,
>                                 dropout=0.5, act=ReLU(), zero_init=True)

*Global Pooling layer that performs a weighted average along the temporal axis

It can be considered as a channel-wise form of local temporal attention. Inspired by the paper: Hyun, J., Seong, H., & Kim, E. (2019). Universal Pooling–A New Pooling Method for Convolutional Neural Networks. arXiv preprint arXiv:1907.11440.*

* * *

source

### GAWP1d

> 
>      GAWP1d (n_in, seq_len, n_layers=2, ln=False, dropout=0.5, act=ReLU(),
>              zero_init=False)

_Global AdaptiveWeightedAvgPool1d + Flatten_

* * *

source

### GACP1d

> 
>      GACP1d (output_size=1)

_Global AdaptiveConcatPool + Flatten_

* * *

source

### GAP1d

> 
>      GAP1d (output_size=1)

_Global Adaptive Pooling + Flatten_

* * *

source

### gwa_pool_head

> 
>      gwa_pool_head (n_in, c_out, seq_len, bn=True, fc_dropout=0.0)

* * *

source

### GlobalWeightedAveragePool1d

> 
>      GlobalWeightedAveragePool1d (n_in, seq_len)

*Global Weighted Average Pooling layer

Inspired by Building Efficient CNN Architecture for Offline Handwritten Chinese Character Recognition https://arxiv.org/pdf/1804.01259.pdf*
    
    
    t = torch.randn(16, 64, 50)
    head = gwa_pool_head(64, 5, 50)
    test_eq(head(t).shape, (16, 5))__

* * *

source

### attentional_pool_head

> 
>      attentional_pool_head (n_in, c_out, seq_len=None, bn=True, **kwargs)

* * *

source

### GAttP1d

> 
>      GAttP1d (n_in, c_out, bn=False)

*A sequential container.

Modules will be added to it in the order they are passed in the constructor. Alternatively, an `OrderedDict` of modules can be passed in. The `forward()` method of `[`Sequential`](https://timeseriesAI.github.io/models.layers.html#sequential)` accepts any input and forwards it to the first module it contains. It then “chains” outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.

The value a `[`Sequential`](https://timeseriesAI.github.io/models.layers.html#sequential)` provides over manually calling a sequence of modules is that it allows treating the whole container as a single module, such that performing a transformation on the `[`Sequential`](https://timeseriesAI.github.io/models.layers.html#sequential)` applies to each of the modules it stores (which are each a registered submodule of the `[`Sequential`](https://timeseriesAI.github.io/models.layers.html#sequential)`).

What’s the difference between a `[`Sequential`](https://timeseriesAI.github.io/models.layers.html#sequential)` and a :class:`torch.nn.ModuleList`? A `ModuleList` is exactly what it sounds like–a list for storing `Module` s! On the other hand, the layers in a `[`Sequential`](https://timeseriesAI.github.io/models.layers.html#sequential)` are connected in a cascading way.

Example::
    
    
    # Using Sequential to create a small model. When `model` is run,
    # input will first be passed to `Conv2d(1,20,5)`. The output of
    # `Conv2d(1,20,5)` will be used as the input to the first
    # `ReLU`; the output of the first `ReLU` will become the input
    # for `Conv2d(20,64,5)`. Finally, the output of
    # `Conv2d(20,64,5)` will be used as input to the second `ReLU`
    model = nn.Sequential(
              nn.Conv2d(1,20,5),
              nn.ReLU(),
              nn.Conv2d(20,64,5),
              nn.ReLU()
            )
    
    # Using Sequential with OrderedDict. This is functionally the
    # same as the above code
    model = nn.Sequential(OrderedDict([
              ('conv1', nn.Conv2d(1,20,5)),
              ('relu1', nn.ReLU()),
              ('conv2', nn.Conv2d(20,64,5)),
              ('relu2', nn.ReLU())
            ]))*

* * *

source

### AttentionalPool1d

> 
>      AttentionalPool1d (n_in, c_out, bn=False)

_Global Adaptive Pooling layer inspired by Attentional Pooling for Action Recognition https://arxiv.org/abs/1711.01467_
    
    
    bs, c_in, seq_len = 16, 1, 50
    c_out = 3
    t = torch.rand(bs, c_in, seq_len)
    test_eq(GAP1d()(t).shape, (bs, c_in))
    test_eq(GACP1d()(t).shape, (bs, c_in*2))
    bs, c_in, seq_len = 16, 4, 50
    t = torch.rand(bs, c_in, seq_len)
    test_eq(GAP1d()(t).shape, (bs, c_in))
    test_eq(GACP1d()(t).shape, (bs, c_in*2))
    test_eq(GAWP1d(c_in, seq_len, n_layers=2, ln=False, dropout=0.5, act=nn.ReLU(), zero_init=False)(t).shape, (bs, c_in))
    test_eq(GAWP1d(c_in, seq_len, n_layers=2, ln=False, dropout=0.5, act=nn.ReLU(), zero_init=False)(t).shape, (bs, c_in))
    test_eq(GAWP1d(c_in, seq_len, n_layers=1, ln=False, dropout=0.5, zero_init=False)(t).shape, (bs, c_in))
    test_eq(GAWP1d(c_in, seq_len, n_layers=1, ln=False, dropout=0.5, zero_init=True)(t).shape, (bs, c_in))
    test_eq(AttentionalPool1d(c_in, c_out)(t).shape, (bs, c_out, 1))__
    
    
    bs, c_in, seq_len = 16, 128, 50
    c_out = 14
    t = torch.rand(bs, c_in, seq_len)
    attp = attentional_pool_head(c_in, c_out)
    test_eq(attp(t).shape, (bs, c_out))__

* * *

source

### PoolingLayer

> 
>      PoolingLayer (method='cls', seq_len=None, token=True, seq_last=True)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    t = torch.arange(24).reshape(2, 3, 4).float()
    test_eq(PoolingLayer('cls', token=True, seq_last=True)(t), tensor([[ 0.,  4.,  8.], [12., 16., 20.]]))
    test_eq(PoolingLayer('max', token=True, seq_last=True)(t), tensor([[ 3.,  7., 11.], [15., 19., 23.]]))
    test_close(PoolingLayer('mean', token=True, seq_last=True)(t), tensor([[ 2.,  6., 10.], [14., 18., 22.]]))
    test_close(PoolingLayer('max-mean', token=True, seq_last=True)(t), tensor([[ 3.,  7., 11.,  2.,  6., 10.],
                                                                               [15., 19., 23., 14., 18., 22.]]))
    test_close(PoolingLayer('flatten', token=True, seq_last=True)(t), tensor([[ 1.,  2.,  3.,  5.,  6.,  7.,  9., 10., 11.],
                                                                              [13., 14., 15., 17., 18., 19., 21., 22., 23.]]))
    test_eq(PoolingLayer('linear', seq_len=4, token=True, seq_last=True)(t).shape, (2, 3))
    test_eq(PoolingLayer('max', token=False, seq_last=True)(t), tensor([[ 3.,  7., 11.], [15., 19., 23.]]))
    test_close(PoolingLayer('mean', token=False, seq_last=True)(t), tensor([[ 1.5000,  5.5000,  9.5000],
                                                                            [13.5000, 17.5000, 21.5000]]))
    test_close(PoolingLayer('max-mean', token=False, seq_last=True)(t), tensor([[ 3.,  7., 11.,  1.5000,  5.5000,  9.5000],
                                                                                [15., 19., 23., 13.5000, 17.5000, 21.5000]]))
    test_close(PoolingLayer('flatten', token=False, seq_last=True)(t), tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.],
                                                                               [12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.]]))
    test_eq(PoolingLayer('linear', seq_len=4, token=False, seq_last=True)(t).shape, (2, 3))__
    
    
    t = torch.arange(24).reshape(2, 3, 4).swapaxes(1,2).float()
    test_eq(PoolingLayer('cls', token=True, seq_last=False)(t), tensor([[ 0.,  4.,  8.], [12., 16., 20.]]))
    test_eq(PoolingLayer('max', token=True, seq_last=False)(t), tensor([[ 3.,  7., 11.], [15., 19., 23.]]))
    test_close(PoolingLayer('mean', token=True, seq_last=False)(t), tensor([[ 2.,  6., 10.], [14., 18., 22.]]))
    test_close(PoolingLayer('max-mean', token=True, seq_last=False)(t), tensor([[ 3.,  7., 11.,  2.,  6., 10.],
                                                                               [15., 19., 23., 14., 18., 22.]]))
    test_close(PoolingLayer('flatten', token=True, seq_last=False)(t), tensor([[ 1.,  5.,  9.,  2.,  6., 10.,  3.,  7., 11.],
                                                                               [13., 17., 21., 14., 18., 22., 15., 19., 23.]]))
    t = torch.arange(24).reshape(2, 3, 4).swapaxes(1,2).float()
    test_eq(PoolingLayer('conv1d', seq_len=4, token=False, seq_last=False)(t).shape, (2, 3))
    test_eq(PoolingLayer('max', token=False, seq_last=False)(t), tensor([[ 3.,  7., 11.], [15., 19., 23.]]))
    test_close(PoolingLayer('mean', token=False, seq_last=False)(t), tensor([[ 1.5000,  5.5000,  9.5000],
                                                                            [13.5000, 17.5000, 21.5000]]))
    test_close(PoolingLayer('max-mean', token=False, seq_last=False)(t), tensor([[ 3.,  7., 11.,  1.5000,  5.5000,  9.5000],
                                                                                [15., 19., 23., 13.5000, 17.5000, 21.5000]]))
    test_close(PoolingLayer('flatten', token=False, seq_last=False)(t), tensor([[ 0.,  4.,  8.,  1.,  5.,  9.,  2.,  6., 10.,  3.,  7., 11.],
                                                                                [12., 16., 20., 13., 17., 21., 14., 18., 22., 15., 19., 23.]]))
    test_eq(PoolingLayer('conv1d', seq_len=4, token=False, seq_last=False)(t).shape, (2, 3))__

* * *

source

### ReGLU

> 
>      ReGLU ()

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### GEGLU

> 
>      GEGLU ()

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### get_act_fn

> 
>      get_act_fn (act, **act_kwargs)
    
    
    test_eq(get_act_fn(nn.ReLU).__repr__(), "ReLU()")
    test_eq(get_act_fn(nn.ReLU()).__repr__(), "ReLU()")
    test_eq(get_act_fn(nn.LeakyReLU, negative_slope=0.05).__repr__(), "LeakyReLU(negative_slope=0.05)")
    test_eq(get_act_fn('reglu').__repr__(), "ReGLU()")
    test_eq(get_act_fn('leakyrelu', negative_slope=0.05).__repr__(), "LeakyReLU(negative_slope=0.05)")__

* * *

source

### RevIN

> 
>      RevIN (c_in:int, affine:bool=True, subtract_last:bool=False, dim:int=2,
>             eps:float=1e-05)

*Reversible Instance Normalization layer adapted from

Kim, T., Kim, J., Tae, Y., Park, C., Choi, J. H., & Choo, J. (2021, September). Reversible instance normalization for accurate time-series forecasting against distribution shift. In International Conference on Learning Representations. Original code: https://github.com/ts-kim/RevIN*

| **Type** | **Default** | **Details**  
---|---|---|---  
c_in | int |  | #features (aka variables or channels)  
affine | bool | True | flag to incidate if RevIN has learnable weight and bias  
subtract_last | bool | False |   
dim | int | 2 | int or tuple of dimensions used to calculate mean and std  
eps | float | 1e-05 | epsilon - parameter added for numerical stability  
  
* * *

source

### RevIN

> 
>      RevIN (c_in:int, affine:bool=True, subtract_last:bool=False, dim:int=2,
>             eps:float=1e-05)

*Reversible Instance Normalization layer adapted from

Kim, T., Kim, J., Tae, Y., Park, C., Choi, J. H., & Choo, J. (2021, September). Reversible instance normalization for accurate time-series forecasting against distribution shift. In International Conference on Learning Representations. Original code: https://github.com/ts-kim/RevIN*

| **Type** | **Default** | **Details**  
---|---|---|---  
c_in | int |  | #features (aka variables or channels)  
affine | bool | True | flag to incidate if RevIN has learnable weight and bias  
subtract_last | bool | False |   
dim | int | 2 | int or tuple of dimensions used to calculate mean and std  
eps | float | 1e-05 | epsilon - parameter added for numerical stability  
      
    
    t = ((torch.rand(16, 5, 100) - .25) * torch.Tensor([.01, .1, 1, 10, 100]).reshape(1, -1, 1)).cumsum(-1)
    t_clone = t.clone()
    l = RevIN(5)
    t_norm = l(t, torch.tensor(True))
    t_denorm = l(t_norm, torch.tensor(False))
    test_close(t_clone, t_denorm, eps=1e-3)__
    
    
    model = RevIN(5, affine=True)
    try:
        scripted_model = torch.jit.script(model)
        file_path = f"test_scripted_model.pt"
        torch.jit.save(scripted_model, file_path)
        scripted_model = torch.jit.load(file_path)
    
        inp = ((torch.rand(16, 5, 100) - .25) * torch.Tensor([.01, .1, 1, 10, 100]).reshape(1, -1, 1)).cumsum(-1)
        normed_output = model(inp, torch.tensor(True))
        demormed_output = model(normed_output, torch.tensor(False))
        scripted_normed_output = scripted_model(inp, torch.tensor(True))
        scripted_denormed_output = scripted_model(scripted_normed_output, torch.tensor(False))
        test_close(normed_output, scripted_normed_output)
        test_close(demormed_output, scripted_denormed_output)
        os.remove(file_path)
        del scripted_model
        gc.collect()
        print('scripting ok')
    except Exception as e:
        print(f'scripting failed: {e}')__
    
    
    scripting ok

* * *

source

### create_pool_head

> 
>      create_pool_head (n_in, c_out, seq_len=None, concat_pool=False,
>                        fc_dropout=0.0, bn=False, y_range=None, **kwargs)
    
    
    bs = 16
    nf = 12
    c_out = 2
    seq_len = 20
    t = torch.rand(bs, nf, seq_len)
    test_eq(create_pool_head(nf, c_out, seq_len, fc_dropout=0.5)(t).shape, (bs, c_out))
    test_eq(create_pool_head(nf, c_out, seq_len, concat_pool=True, fc_dropout=0.5)(t).shape, (bs, c_out))
    create_pool_head(nf, c_out, seq_len, concat_pool=True, bn=True, fc_dropout=.5)__
    
    
    Sequential(
      (0): GACP1d(
        (gacp): AdaptiveConcatPool1d(
          (ap): AdaptiveAvgPool1d(output_size=1)
          (mp): AdaptiveMaxPool1d(output_size=1)
        )
        (flatten): Reshape(bs)
      )
      (1): LinBnDrop(
        (0): BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Dropout(p=0.5, inplace=False)
        (2): Linear(in_features=24, out_features=2, bias=False)
      )
    )

* * *

source

### max_pool_head

> 
>      max_pool_head (n_in, c_out, seq_len, fc_dropout=0.0, bn=False,
>                     y_range=None, **kwargs)
    
    
    bs = 16
    nf = 12
    c_out = 2
    seq_len = 20
    t = torch.rand(bs, nf, seq_len)
    test_eq(max_pool_head(nf, c_out, seq_len, fc_dropout=0.5)(t).shape, (bs, c_out))__

* * *

source

### create_pool_plus_head

> 
>      create_pool_plus_head (*args, lin_ftrs=None, fc_dropout=0.0,
>                             concat_pool=True, bn_final=False, lin_first=False,
>                             y_range=None)
    
    
    bs = 16
    nf = 12
    c_out = 2
    seq_len = 20
    t = torch.rand(bs, nf, seq_len)
    test_eq(create_pool_plus_head(nf, c_out, seq_len, fc_dropout=0.5)(t).shape, (bs, c_out))
    test_eq(create_pool_plus_head(nf, c_out, concat_pool=True, fc_dropout=0.5)(t).shape, (bs, c_out))
    create_pool_plus_head(nf, c_out, seq_len, fc_dropout=0.5)__
    
    
    Sequential(
      (0): AdaptiveConcatPool1d(
        (ap): AdaptiveAvgPool1d(output_size=1)
        (mp): AdaptiveMaxPool1d(output_size=1)
      )
      (1): Reshape(bs)
      (2): BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): Dropout(p=0.25, inplace=False)
      (4): Linear(in_features=24, out_features=512, bias=False)
      (5): ReLU(inplace=True)
      (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (7): Dropout(p=0.5, inplace=False)
      (8): Linear(in_features=512, out_features=2, bias=False)
    )

* * *

source

### create_conv_head

> 
>      create_conv_head (*args, adaptive_size=None, y_range=None)
    
    
    bs = 16
    nf = 12
    c_out = 2
    seq_len = 20
    t = torch.rand(bs, nf, seq_len)
    test_eq(create_conv_head(nf, c_out, seq_len)(t).shape, (bs, c_out))
    test_eq(create_conv_head(nf, c_out, adaptive_size=50)(t).shape, (bs, c_out))
    create_conv_head(nf, c_out, 50)__
    
    
    Sequential(
      (0): ConvBlock(
        (0): Conv1d(12, 6, kernel_size=(1,), stride=(1,), bias=False)
        (1): BatchNorm1d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (1): ConvBlock(
        (0): Conv1d(6, 3, kernel_size=(1,), stride=(1,), bias=False)
        (1): BatchNorm1d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (2): ConvBlock(
        (0): Conv1d(3, 2, kernel_size=(1,), stride=(1,), bias=False)
        (1): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (3): GAP1d(
        (gap): AdaptiveAvgPool1d(output_size=1)
        (flatten): Reshape(bs)
      )
    )

* * *

source

### create_mlp_head

> 
>      create_mlp_head (nf, c_out, seq_len=None, flatten=True, fc_dropout=0.0,
>                       bn=False, lin_first=False, y_range=None)
    
    
    bs = 16
    nf = 12
    c_out = 2
    seq_len = 20
    t = torch.rand(bs, nf, seq_len)
    test_eq(create_mlp_head(nf, c_out, seq_len, fc_dropout=0.5)(t).shape, (bs, c_out))
    t = torch.rand(bs, nf, seq_len)
    create_mlp_head(nf, c_out, seq_len, bn=True, fc_dropout=.5)__
    
    
    Sequential(
      (0): Reshape(bs)
      (1): LinBnDrop(
        (0): BatchNorm1d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Dropout(p=0.5, inplace=False)
        (2): Linear(in_features=240, out_features=2, bias=False)
      )
    )

* * *

source

### create_fc_head

> 
>      create_fc_head (nf, c_out, seq_len=None, flatten=True, lin_ftrs=None,
>                      y_range=None, fc_dropout=0.0, bn=False, bn_final=False,
>                      act=ReLU(inplace=True))
    
    
    bs = 16
    nf = 12
    c_out = 2
    seq_len = 20
    t = torch.rand(bs, nf, seq_len)
    test_eq(create_fc_head(nf, c_out, seq_len, fc_dropout=0.5)(t).shape, (bs, c_out))
    create_mlp_head(nf, c_out, seq_len, bn=True, fc_dropout=.5)__
    
    
    Sequential(
      (0): Reshape(bs)
      (1): LinBnDrop(
        (0): BatchNorm1d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Dropout(p=0.5, inplace=False)
        (2): Linear(in_features=240, out_features=2, bias=False)
      )
    )

* * *

source

### create_rnn_head

> 
>      create_rnn_head (*args, fc_dropout=0.0, bn=False, y_range=None)
    
    
    bs = 16
    nf = 12
    c_out = 2
    seq_len = 20
    t = torch.rand(bs, nf, seq_len)
    test_eq(create_rnn_head(nf, c_out, seq_len, fc_dropout=0.5)(t).shape, (bs, c_out))
    create_rnn_head(nf, c_out, seq_len, bn=True, fc_dropout=.5)__
    
    
    Sequential(
      (0): LastStep()
      (1): LinBnDrop(
        (0): BatchNorm1d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Dropout(p=0.5, inplace=False)
        (2): Linear(in_features=12, out_features=2, bias=False)
      )
    )

* * *

source

### imputation_head

> 
>      imputation_head (c_in, c_out, seq_len=None, ks=1, y_range=None,
>                       fc_dropout=0.0)
    
    
    bs = 16
    nf = 12
    ni = 2
    seq_len = 20
    t = torch.rand(bs, nf, seq_len)
    head = imputation_head(nf, ni, seq_len=None, ks=1, y_range=None, fc_dropout=0.)
    test_eq(head(t).shape, (bs, ni, seq_len))
    head = imputation_head(nf, ni, seq_len=None, ks=1, y_range=(.3,.7), fc_dropout=0.)
    test_ge(head(t).min(), .3)
    test_le(head(t).max(), .7)
    y_range = (tensor([0.1000, 0.1000, 0.1000, 0.1000, 0.2000, 0.2000, 0.2000, 0.2000, 0.3000,
                       0.3000, 0.3000, 0.3000]),
               tensor([0.6000, 0.6000, 0.6000, 0.6000, 0.7000, 0.7000, 0.7000, 0.7000, 0.8000,
                       0.8000, 0.8000, 0.8000]))
    test_ge(head(t).min(), .1)
    test_le(head(t).max(), .9)
    head = imputation_head(nf, ni, seq_len=None, ks=1, y_range=y_range, fc_dropout=0.)
    head __
    
    
    Sequential(
      (0): Dropout(p=0.0, inplace=False)
      (1): Conv1d(12, 2, kernel_size=(1,), stride=(1,))
      (2): fastai.layers.SigmoidRange(low=tensor([0.1000, 0.1000, 0.1000, 0.1000, 0.2000, 0.2000, 0.2000, 0.2000, 0.3000,
              0.3000, 0.3000, 0.3000]), high=tensor([0.6000, 0.6000, 0.6000, 0.6000, 0.7000, 0.7000, 0.7000, 0.7000, 0.8000,
              0.8000, 0.8000, 0.8000]))
    )

* * *

source

### create_conv_lin_nd_head

> 
>      create_conv_lin_nd_head (n_in, n_out, seq_len, d, conv_first=True,
>                               conv_bn=False, lin_bn=False, fc_dropout=0.0,
>                               **kwargs)

_Module to create a nd output head_
    
    
    bs = 16
    nf = 32
    c = 5
    seq_len = 10
    d = 2
    targ = torch.randint(0, c, (bs,d))
    t = torch.randn(bs, nf, seq_len)
    head = conv_lin_nd_head(nf, c, seq_len, d, conv_first=True, fc_dropout=.5)
    inp = head(t)
    test_eq(inp.shape, (bs, d, c))
    loss = CrossEntropyLossFlat()(inp, targ)
    loss, head __
    
    
    (TensorBase(1.7252, grad_fn=<AliasBackward0>),
     create_conv_lin_nd_head(
       (0): Conv1d(32, 5, kernel_size=(1,), stride=(1,))
       (1): Dropout(p=0.5, inplace=False)
       (2): Linear(in_features=10, out_features=2, bias=True)
       (3): Transpose(dims=-1, -2).contiguous()
       (4): Reshape(bs, 2, 5)
     ))
    
    
    bs = 16
    nf = 32
    c = 5
    seq_len = 10
    d = [2, 8]
    targ = torch.randint(0, c, [bs]+d)
    t = torch.randn(bs, nf, seq_len)
    head = conv_lin_nd_head(nf, c, seq_len, d, conv_first=False, fc_dropout=.5)
    inp = head(t)
    test_eq(inp.shape, [bs]+d+[c])
    loss = CrossEntropyLossFlat()(inp, targ)
    loss, head __
    
    
    (TensorBase(1.6647, grad_fn=<AliasBackward0>),
     create_conv_lin_nd_head(
       (0): Dropout(p=0.5, inplace=False)
       (1): Linear(in_features=10, out_features=16, bias=True)
       (2): Conv1d(32, 5, kernel_size=(1,), stride=(1,))
       (3): Transpose(dims=-1, -2).contiguous()
       (4): Reshape(bs, 2, 8, 5)
     ))
    
    
    bs = 16
    nf = 32
    c = 1
    seq_len = 10
    d = 2
    targ = torch.rand(bs, d)
    t = torch.randn(bs, nf, seq_len)
    head = conv_lin_nd_head(nf, c, seq_len, d, conv_first=False, fc_dropout=.5)
    inp = head(t)
    test_eq(inp.shape, (bs, d))
    loss = L1LossFlat()(inp, targ)
    loss, head __
    
    
    (TensorBase(0.7063, grad_fn=<AliasBackward0>),
     create_conv_lin_nd_head(
       (0): Dropout(p=0.5, inplace=False)
       (1): Linear(in_features=10, out_features=2, bias=True)
       (2): Conv1d(32, 1, kernel_size=(1,), stride=(1,))
       (3): Transpose(dims=-1, -2).contiguous()
       (4): Reshape(bs, 2)
     ))
    
    
    bs = 16
    nf = 32
    c = 1
    seq_len = 10
    d = [2,3]
    targ = torch.rand(bs, *d)
    t = torch.randn(bs, nf, seq_len)
    head = conv_lin_nd_head(nf, c, seq_len, d, conv_first=False, fc_dropout=.5)
    inp = head(t)
    test_eq(inp.shape, [bs]+d)
    loss = L1LossFlat()(inp, targ)
    loss, head __
    
    
    (TensorBase(0.6203, grad_fn=<AliasBackward0>),
     create_conv_lin_nd_head(
       (0): Dropout(p=0.5, inplace=False)
       (1): Linear(in_features=10, out_features=6, bias=True)
       (2): Conv1d(32, 1, kernel_size=(1,), stride=(1,))
       (3): Transpose(dims=-1, -2).contiguous()
       (4): Reshape(bs, 2, 3)
     ))

* * *

source

### lin_nd_head

> 
>      lin_nd_head (n_in, n_out, seq_len=None, d=None, flatten=False,
>                   use_bn=False, fc_dropout=0.0)

_Module to create a nd output head with linear layers_
    
    
    bs = 16
    nf = 32
    seq_len = 50
    x = torch.normal(0, 1, (bs, nf, seq_len))
    
    for use_bn in [False, True]:
        for fc_dropout in [0, 0.2]:
            for flatten in [False, True]:
                for c in [1, 3]:
                    for d in [None, (50,), (50,10), (30,5), (50,2,3), (30,2,3)]:
                        for q_len in [1, seq_len]:
                            head = lin_nd_head(nf, c, q_len, d, flatten=flatten, use_bn=use_bn, fc_dropout=fc_dropout)
                            test_eq(head(x).shape, (bs, ) + (d if d is not None else ()) + ((c,) if c != 1 else ()))__
    
    
    bs = 16
    nf = 32
    c = 5
    seq_len = 10
    d = 2
    targ = torch.randint(0, c, (bs,d))
    t = torch.randn(bs, nf, seq_len)
    head = lin_nd_head(nf, c, seq_len, d, fc_dropout=.5)
    inp = head(t)
    test_eq(inp.shape, (bs, d, c))
    loss = CrossEntropyLossFlat()(inp, targ)
    loss, head __
    
    
    (TensorBase(1.7711, grad_fn=<AliasBackward0>),
     lin_nd_head(
       (0): Dropout(p=0.5, inplace=False)
       (1): Reshape(bs)
       (2): Linear(in_features=320, out_features=10, bias=True)
       (3): Reshape(bs, 2, 5)
     ))
    
    
    bs = 16
    nf = 32
    c = 5
    seq_len = 10
    d = [2, 8]
    targ = torch.randint(0, c, [bs]+d)
    t = torch.randn(bs, nf, seq_len)
    head = lin_nd_head(nf, c, seq_len, d, fc_dropout=.5)
    inp = head(t)
    test_eq(inp.shape, [bs]+d+[c])
    loss = CrossEntropyLossFlat()(inp, targ)
    loss, head __
    
    
    (TensorBase(1.8884, grad_fn=<AliasBackward0>),
     lin_nd_head(
       (0): Dropout(p=0.5, inplace=False)
       (1): Reshape(bs)
       (2): Linear(in_features=320, out_features=80, bias=True)
       (3): Reshape(bs, 2, 8, 5)
     ))
    
    
    bs = 16
    nf = 32
    c = 1
    seq_len = 10
    d = 2
    targ = torch.rand(bs, d)
    t = torch.randn(bs, nf, seq_len)
    head = lin_nd_head(nf, c, seq_len, d, fc_dropout=.5)
    inp = head(t)
    test_eq(inp.shape, (bs, d))
    loss = L1LossFlat()(inp, targ)
    loss, head __
    
    
    (TensorBase(0.7737, grad_fn=<AliasBackward0>),
     lin_nd_head(
       (0): Dropout(p=0.5, inplace=False)
       (1): Reshape(bs)
       (2): Linear(in_features=320, out_features=2, bias=True)
       (3): Reshape(bs, 2)
     ))
    
    
    bs = 16
    nf = 32
    c = 1
    seq_len = 10
    d = [2,3]
    targ = torch.rand(bs, *d)
    t = torch.randn(bs, nf, seq_len)
    head = lin_nd_head(nf, c, seq_len, d, fc_dropout=.5)
    inp = head(t)
    test_eq(inp.shape, [bs]+d)
    loss = L1LossFlat()(inp, targ)
    loss, head __
    
    
    (TensorBase(0.8873, grad_fn=<AliasBackward0>),
     lin_nd_head(
       (0): Dropout(p=0.5, inplace=False)
       (1): Reshape(bs)
       (2): Linear(in_features=320, out_features=6, bias=True)
       (3): Reshape(bs, 2, 3)
     ))

* * *

source

### rocket_nd_head

> 
>      rocket_nd_head (n_in, n_out, seq_len=None, d=None, use_bn=False,
>                      fc_dropout=0.0, zero_init=True)

_Module to create a nd output head with linear layers for the rocket family of models_
    
    
    bs = 16
    nf = 99
    seq_len = 1
    x = torch.normal(0, 1, (bs, nf, seq_len))
    
    for use_bn in [False, True]:
        for fc_dropout in [0, 0.2]:
            for c in [1, 3]:
                for d in [None, (50,), (50,10), (30,5), (50,2,3), (30,2,3)]:
                    head = rocket_nd_head(nf, c, 1, d, use_bn=use_bn, fc_dropout=fc_dropout)
                    test_eq(head(x).shape, (bs, ) + (d if d is not None else ()) + ((c,) if c != 1 else ()))__

* * *

source

### xresnet1d_nd_head

> 
>      xresnet1d_nd_head (n_in, n_out, seq_len=None, d=None, use_bn=False,
>                         fc_dropout=0.0, zero_init=True)

_Module to create a nd output head with linear layers for the xresnet family of models_
    
    
    bs = 16
    nf = 99
    seq_len = 2
    x = torch.normal(0, 1, (bs, nf, seq_len))
    
    for use_bn in [False, True]:
        for fc_dropout in [0, 0.2]:
            for c in [1, 3]:
                for d in [None, (50,), (50,10), (30,5), (50,2,3), (30,2,3)]:
                    head = xresnet1d_nd_head(nf, c, 1, d, use_bn=use_bn, fc_dropout=fc_dropout)
                    test_eq(head(x).shape, (bs, ) + (d if d is not None else ()) + ((c,) if c != 1 else ()))__

* * *

source

### create_conv_3d_head

> 
>      create_conv_3d_head (n_in, n_out, seq_len, d, use_bn=False, **kwargs)

_Module to create a nd output head with a convolutional layer_
    
    
    bs = 16
    nf = 32
    c = 5
    seq_len = 10
    d = 10
    targ = torch.randint(0, c, (bs,d))
    t = torch.randn(bs, nf, seq_len)
    head = conv_3d_head(nf, c, seq_len, d)
    inp = head(t)
    test_eq(inp.shape, (bs, d, c))
    loss = CrossEntropyLossFlat()(inp, targ)
    loss, head __
    
    
    (TensorBase(1.8352, grad_fn=<AliasBackward0>),
     create_conv_3d_head(
       (0): ConvBlock(
         (0): Conv1d(32, 5, kernel_size=(1,), stride=(1,))
       )
       (1): Transpose(dims=-1, -2).contiguous()
     ))
    
    
    bs = 16
    nf = 32
    c = 1
    seq_len = 10
    d = 10
    targ = torch.rand(bs, d)
    t = torch.randn(bs, nf, seq_len)
    head = conv_3d_head(nf, c, seq_len, d)
    inp = head(t)
    test_eq(inp.shape, (bs, d))
    loss = L1LossFlat()(inp, targ)
    loss, head __
    
    
    (TensorBase(0.6711, grad_fn=<AliasBackward0>),
     create_conv_3d_head(
       (0): ConvBlock(
         (0): Conv1d(32, 1, kernel_size=(1,), stride=(1,))
       )
       (1): Transpose(dims=-1, -2).contiguous()
       (2): Squeeze(dim=-1)
     ))

* * *

source

### universal_pool_head

> 
>      universal_pool_head (n_in, c_out, seq_len, mult=2, pool_n_layers=2,
>                           pool_ln=True, pool_dropout=0.5, pool_act=ReLU(),
>                           zero_init=True, bn=True, fc_dropout=0.0)
    
    
    bs, c_in, seq_len = 16, 128, 50
    c_out = 14
    t = torch.rand(bs, c_in, seq_len)
    uph = universal_pool_head(c_in, c_out, seq_len)
    test_eq(uph(t).shape, (bs, c_out))
    uph = universal_pool_head(c_in, c_out, seq_len, 2)
    test_eq(uph(t).shape, (bs, c_out))__
    
    
    bs, c_in, seq_len = 16, 128, 50
    c_out = 14
    d = 5
    t = torch.rand(bs, c_in, seq_len)
    for head in heads:
        print(head.__name__)
        if head.__name__ == "create_conv_3d_head":
            h = head(c_in, c_out, seq_len, seq_len)
            test_eq(h(t).shape, (bs, seq_len, c_out))
        elif 'nd' in head.__name__:
            h = head(c_in, c_out, seq_len, d)
            test_eq(h(t).shape, (bs, d, c_out))
        else:
            h = head(c_in, c_out, seq_len)
            test_eq(h(t).shape, (bs, c_out))__
    
    
    create_mlp_head
    create_fc_head
    average_pool_head
    max_pool_head
    concat_pool_head
    create_pool_plus_head
    create_conv_head
    create_rnn_head
    create_conv_lin_nd_head
    lin_nd_head
    create_conv_3d_head
    attentional_pool_head
    universal_pool_head
    gwa_pool_head

* * *

source

### SqueezeExciteBlock

> 
>      SqueezeExciteBlock (ni, reduction=16)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    bs = 2
    ni = 32
    sl = 4
    t = torch.rand(bs, ni, sl)
    test_eq(SqueezeExciteBlock(ni)(t).shape, (bs, ni, sl))__

* * *

source

### GaussianNoise

> 
>      GaussianNoise (sigma=0.1, is_relative_detach=True)

*Gaussian noise regularizer.

Args: sigma (float, optional): relative standard deviation used to generate the noise. Relative means that it will be multiplied by the magnitude of the value your are adding the noise to. This means that sigma can be the same regardless of the scale of the vector. is_relative_detach (bool, optional): whether to detach the variable before computing the scale of the noise. If `False` then the scale of the noise won’t be seen as a constant but something to optimize: this will bias the network to generate vectors with smaller values.*
    
    
    t = torch.ones(2,3,4)
    test_ne(GaussianNoise()(t), t)
    test_eq(GaussianNoise()(t).shape, t.shape)
    t = torch.ones(2,3)
    test_ne(GaussianNoise()(t), t)
    test_eq(GaussianNoise()(t).shape, t.shape)
    t = torch.ones(2)
    test_ne(GaussianNoise()(t), t)
    test_eq(GaussianNoise()(t).shape, t.shape)__

* * *

source

### TokenLayer

> 
>      TokenLayer (token=True)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### PositionwiseFeedForward

> 
>      PositionwiseFeedForward (dim, dropout=0.0, act='reglu', mlp_ratio=1)

*A sequential container.

Modules will be added to it in the order they are passed in the constructor. Alternatively, an `OrderedDict` of modules can be passed in. The `forward()` method of `[`Sequential`](https://timeseriesAI.github.io/models.layers.html#sequential)` accepts any input and forwards it to the first module it contains. It then “chains” outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.

The value a `[`Sequential`](https://timeseriesAI.github.io/models.layers.html#sequential)` provides over manually calling a sequence of modules is that it allows treating the whole container as a single module, such that performing a transformation on the `[`Sequential`](https://timeseriesAI.github.io/models.layers.html#sequential)` applies to each of the modules it stores (which are each a registered submodule of the `[`Sequential`](https://timeseriesAI.github.io/models.layers.html#sequential)`).

What’s the difference between a `[`Sequential`](https://timeseriesAI.github.io/models.layers.html#sequential)` and a :class:`torch.nn.ModuleList`? A `ModuleList` is exactly what it sounds like–a list for storing `Module` s! On the other hand, the layers in a `[`Sequential`](https://timeseriesAI.github.io/models.layers.html#sequential)` are connected in a cascading way.

Example::
    
    
    # Using Sequential to create a small model. When `model` is run,
    # input will first be passed to `Conv2d(1,20,5)`. The output of
    # `Conv2d(1,20,5)` will be used as the input to the first
    # `ReLU`; the output of the first `ReLU` will become the input
    # for `Conv2d(20,64,5)`. Finally, the output of
    # `Conv2d(20,64,5)` will be used as input to the second `ReLU`
    model = nn.Sequential(
              nn.Conv2d(1,20,5),
              nn.ReLU(),
              nn.Conv2d(20,64,5),
              nn.ReLU()
            )
    
    # Using Sequential with OrderedDict. This is functionally the
    # same as the above code
    model = nn.Sequential(OrderedDict([
              ('conv1', nn.Conv2d(1,20,5)),
              ('relu1', nn.ReLU()),
              ('conv2', nn.Conv2d(20,64,5)),
              ('relu2', nn.ReLU())
            ]))*
    
    
    t = torch.randn(2,3,10)
    m = PositionwiseFeedForward(10, dropout=0., act='reglu', mlp_ratio=1)
    test_eq(m(t).shape, t.shape)
    m = PositionwiseFeedForward(10, dropout=0., act='smelu', mlp_ratio=1)
    test_eq(m(t).shape, t.shape)__

* * *

source

### ScaledDotProductAttention

> 
>      ScaledDotProductAttention (d_model, n_heads, attn_dropout=0.0,
>                                 res_attention=False, lsa=False)

_Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets by Lee et al, 2021)_
    
    
    B = 16
    C = 10
    M = 1500 # seq_len
    
    n_heads = 1
    D = 128 # model dimension
    N = 512 # max_seq_len - latent's index dimension
    d_k = D // n_heads
    
    xb = torch.randn(B, C, M)
    xb = (xb - xb.mean()) / xb.std()
    
    # Attention
    # input (Q)
    lin = nn.Linear(M, N, bias=False)
    Q = lin(xb).transpose(1,2)
    test_eq(Q.shape, (B, N, C))
    
    # q
    to_q = nn.Linear(C, D, bias=False)
    q = to_q(Q)
    q = nn.LayerNorm(D)(q)
    
    # k, v
    context = xb.transpose(1,2)
    to_kv = nn.Linear(C, D * 2, bias=False)
    k, v = to_kv(context).chunk(2, dim = -1)
    k = k.transpose(-1, -2)
    k = nn.LayerNorm(M)(k)
    v = nn.LayerNorm(D)(v)
    
    test_eq(q.shape, (B, N, D))
    test_eq(k.shape, (B, D, M))
    test_eq(v.shape, (B, M, D))
    
    output, attn, scores = ScaledDotProductAttention(D, n_heads, res_attention=True)(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1))
    test_eq(output.shape, (B, 1, N, D))
    test_eq(attn.shape, (B, 1, N, M))
    test_eq(scores.shape, (B, 1, N, M))
    scores.mean(), scores.std()__
    
    
    (tensor(-2.3159e-10, grad_fn=<MeanBackward0>),
     tensor(0.9743, grad_fn=<StdBackward0>))

* * *

source

### MultiheadAttention

> 
>      MultiheadAttention (d_model, n_heads, d_k=None, d_v=None,
>                          res_attention=False, attn_dropout=0.0,
>                          proj_dropout=0.0, qkv_bias=True, lsa=False)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    q = torch.rand([16, 3, 50, 8])
    k = torch.rand([16, 3, 50, 8]).transpose(-1, -2)
    v = torch.rand([16, 3, 50, 6])
    attn_mask = torch.triu(torch.ones(50, 50)) # shape: q_len x q_len
    key_padding_mask = torch.zeros(16, 50)
    key_padding_mask[[1, 3, 6, 15], -10:] = 1
    key_padding_mask = key_padding_mask.bool()
    print('attn_mask', attn_mask.shape, 'key_padding_mask', key_padding_mask.shape)
    output, attn = ScaledDotProductAttention(24, 3, attn_dropout=.1)(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
    output.shape, attn.shape __
    
    
    attn_mask torch.Size([50, 50]) key_padding_mask torch.Size([16, 50])
    
    
    (torch.Size([16, 3, 50, 6]), torch.Size([16, 3, 50, 50]))
    
    
    t = torch.rand(16, 50, 128)
    output, attn = MultiheadAttention(d_model=128, n_heads=3, d_k=8, d_v=6)(t, t, t, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
    output.shape, attn.shape __
    
    
    (torch.Size([16, 50, 128]), torch.Size([16, 3, 50, 50]))

Test multi-head attention with self-locality attention
    
    
    # lsa (locality self-sttention)
    t = torch.rand(16, 50, 128)
    attn_mask = torch.eye(50).reshape(1, 1, 50, 50).bool()
    output, attn = MultiheadAttention(d_model=128, n_heads=8, lsa=True)(t, t, t, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
    output.shape, attn.shape __
    
    
    (torch.Size([16, 50, 128]), torch.Size([16, 8, 50, 50]))
    
    
    t = torch.rand(16, 50, 128)
    att_mask = (torch.rand((50, 50)) > .85).float()
    att_mask[att_mask == 1] = -np.inf
    
    mha = MultiheadAttention(d_model=128, n_heads=3, d_k=8, d_v=6)
    output, attn = mha(t, t, t, attn_mask=att_mask)
    test_eq(torch.isnan(output).sum().item(), 0)
    test_eq(torch.isnan(attn).sum().item(), 0)
    loss = output[:2, :].sum()
    test_eq(torch.isnan(loss).sum().item(), 0)
    loss.backward()
    for n, p in mha.named_parameters():
        if p.grad is not None:
            test_eq(torch.isnan(p.grad).sum().item(), 0)__
    
    
    t = torch.rand(16, 50, 128)
    attn_mask = (torch.rand((50, 50)) > .85)
    
    # True values will be masked
    mha = MultiheadAttention(d_model=128, n_heads=3, d_k=8, d_v=6)
    output, attn = mha(t, t, t, attn_mask=att_mask)
    test_eq(torch.isnan(output).sum().item(), 0)
    test_eq(torch.isnan(attn).sum().item(), 0)
    loss = output[:2, :].sum()
    test_eq(torch.isnan(loss).sum().item(), 0)
    loss.backward()
    for n, p in mha.named_parameters():
        if p.grad is not None:
            test_eq(torch.isnan(p.grad).sum().item(), 0)__

* * *

source

### MultiConv1d

> 
>      MultiConv1d (ni, nf=None, kss=[1, 3, 5, 7], keep_original=False,
>                   separable=False, dim=1, **kwargs)

_Module that applies multiple convolutions with different kernel sizes_
    
    
    t = torch.rand(16, 6, 37)
    test_eq(MultiConv1d(6, None, kss=[1,3,5], keep_original=True)(t).shape, [16, 24, 37])
    test_eq(MultiConv1d(6, 36, kss=[1,3,5], keep_original=False)(t).shape, [16, 36, 37])
    test_eq(MultiConv1d(6, None, kss=[1,3,5], keep_original=True, dim=-1)(t).shape, [16, 6, 37*4])
    test_eq(MultiConv1d(6, 60, kss=[1,3,5], keep_original=True)(t).shape, [16, 60, 37])
    test_eq(MultiConv1d(6, 60, kss=[1,3,5], separable=True)(t).shape, [16, 60, 37])__

* * *

source

### LSTMOutput

> 
>      LSTMOutput ()

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    t = ([1], [2], [3])
    test_eq(LSTMOutput()(t), [1])__

* * *

source

### emb_sz_rule

> 
>      emb_sz_rule (n_cat)

_Rule of thumb to pick embedding size corresponding to`n_cat` (original from fastai)_
    
    
    test_eq(emb_sz_rule(7), 5)__

* * *

source

### TSEmbedding

> 
>      TSEmbedding (ni, nf, std=0.01, padding_idx=None)

_Embedding layer with truncated normal initialization adapted from fastai_

* * *

source

### MultiEmbedding

> 
>      MultiEmbedding (c_in, n_cat_embeds, cat_embed_dims=None, cat_pos=None,
>                      std=0.01, cat_padding_idxs=None)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    a = alphabet[np.random.randint(0,3,40)]
    b = ALPHABET[np.random.randint(6,10,40)]
    c = np.random.rand(40).reshape(4,1,10)
    map_a = {k:v for v,k in enumerate(np.unique(a))}
    map_b = {k:v for v,k in enumerate(np.unique(b))}
    n_embeds = [len(m.keys()) for m in [map_a, map_b]]
    szs = [emb_sz_rule(n) for n in n_embeds]
    a = np.asarray(a.map(map_a)).reshape(4,1,10)
    b = np.asarray(b.map(map_b)).reshape(4,1,10)
    inp = torch.from_numpy(np.concatenate((c,a,b), 1)).float()
    memb = MultiEmbedding(3, n_embeds, cat_pos=[1,2])
    # registered buffers are part of the state_dict() but not module.parameters()
    assert all([(k in memb.state_dict().keys()) for k in ['cat_pos', 'cont_pos']])
    embeddings = memb(inp)
    print(n_embeds, szs, inp.shape, embeddings.shape)
    test_eq(embeddings.shape, (inp.shape[0],sum(szs)+1,inp.shape[-1]))__
    
    
    [3, 4] [3, 3] torch.Size([4, 3, 10]) torch.Size([4, 7, 10])
    
    
    me = MultiEmbedding(3, 4, cat_pos=2)
    test_eq(me.cat_embed[0].weight.shape, (4,3))
    test_eq(me.cat_pos.cpu().item(), 2)__

  * __Report an issue


