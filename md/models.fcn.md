## On this page

  * FCN



  * __Report an issue



  1. Models
  2. CNNs
  3. FCN



# FCN

This is an unofficial PyTorch implementation created by Ignacio Oguiza (oguiza@timeseriesAI.co) based on:

  * Wang, Z., Yan, W., & Oates, T. (2017, May). **Time series classification from scratch with deep neural networks: A strong baseline**. In 2017 international joint conference on neural networks (IJCNN) (pp. 1578-1585). IEEE.

  * Fawaz, H. I., Forestier, G., Weber, J., Idoumghar, L., & Muller, P. A. (2019). **Deep learning for time series classification: a review**. Data Mining and Knowledge Discovery, 33(4), 917-963.




Official FCN TensorFlow implementation: https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py.

Note: kernel filter size 8 has been replaced by 7 (since we believe it’s a bug).

* * *

source

### FCN

> 
>      FCN (c_in, c_out, layers=[128, 256, 128], kss=[7, 5, 3])

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    bs = 16
    nvars = 3
    seq_len = 128
    c_out = 2
    xb = torch.rand(bs, nvars, seq_len)
    model = FCN(nvars, c_out)
    test_eq(model(xb).shape, (bs, c_out))
    model __
    
    
    FCN(
      (convblock1): ConvBlock(
        (0): Conv1d(3, 128, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (convblock2): ConvBlock(
        (0): Conv1d(128, 256, kernel_size=(5,), stride=(1,), padding=(2,), bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (convblock3): ConvBlock(
        (0): Conv1d(256, 128, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (gap): GAP1d(
        (gap): AdaptiveAvgPool1d(output_size=1)
        (flatten): Flatten(full=False)
      )
      (fc): Linear(in_features=128, out_features=2, bias=True)
    )

  * __Report an issue


