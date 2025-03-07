## On this page

  * MLP



  * __Report an issue



  1. Models
  2. MLPs
  3. MLP



# MLP

This is an unofficial PyTorch implementation created by Ignacio Oguiza (oguiza@timeseriesAI.co) based on:

Fawaz, H. I., Forestier, G., Weber, J., Idoumghar, L., & Muller, P. A. (2019). **Deep learning for time series classification: a review**. Data Mining and Knowledge Discovery, 33(4), 917-963.

Official MLP TensorFlow implementation: https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mlp.py

* * *

source

### MLP

> 
>      MLP (c_in, c_out, seq_len, layers=[500, 500, 500], ps=[0.1, 0.2, 0.2],
>           act=ReLU(inplace=True), use_bn=False, bn_final=False,
>           lin_first=False, fc_dropout=0.0, y_range=None)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    bs = 16
    nvars = 3
    seq_len = 128
    c_out = 2
    xb = torch.rand(bs, nvars, seq_len)
    model = MLP(nvars, c_out, seq_len)
    test_eq(model(xb).shape, (bs, c_out))
    model __
    
    
    MLP(
      (flatten): Reshape(bs)
      (mlp): ModuleList(
        (0): LinBnDrop(
          (0): Dropout(p=0.1, inplace=False)
          (1): Linear(in_features=384, out_features=500, bias=True)
          (2): ReLU(inplace=True)
        )
        (1): LinBnDrop(
          (0): Dropout(p=0.2, inplace=False)
          (1): Linear(in_features=500, out_features=500, bias=True)
          (2): ReLU(inplace=True)
        )
        (2): LinBnDrop(
          (0): Dropout(p=0.2, inplace=False)
          (1): Linear(in_features=500, out_features=500, bias=True)
          (2): ReLU(inplace=True)
        )
      )
      (head): Sequential(
        (0): LinBnDrop(
          (0): Linear(in_features=500, out_features=2, bias=True)
        )
      )
    )

  * __Report an issue


