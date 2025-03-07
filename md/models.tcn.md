## On this page

  * TCN
  * TemporalConvNet
  * TemporalBlock



  * __Report an issue



  1. Models
  2. CNNs
  3. TCN



# TCN

This is an unofficial PyTorch implementation by Ignacio Oguiza (oguiza@timeseriesAI.co) based on:

  * Bai, S., Kolter, J. Z., & Koltun, V. (2018). **An empirical evaluation of generic convolutional and recurrent networks for sequence modeling**. arXiv preprint arXiv:1803.01271.
  * Official TCN PyTorch implementation: https://github.com/locuslab/TCN



* * *

source

### TCN

> 
>      TCN (c_in, c_out, layers=[25, 25, 25, 25, 25, 25, 25, 25], ks=7,
>           conv_dropout=0.0, fc_dropout=0.0)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### TemporalConvNet

> 
>      TemporalConvNet (c_in, layers, ks=2, dropout=0.0)

* * *

source

### TemporalBlock

> 
>      TemporalBlock (ni, nf, ks, stride, dilation, padding, dropout=0.0)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    bs = 16
    nvars = 3
    seq_len = 128
    c_out = 2
    xb = torch.rand(bs, nvars, seq_len)
    model = TCN(nvars, c_out, fc_dropout=.5)
    test_eq(model(xb).shape, (bs, c_out))
    model = TCN(nvars, c_out, conv_dropout=.2)
    test_eq(model(xb).shape, (bs, c_out))
    model = TCN(nvars, c_out)
    test_eq(model(xb).shape, (bs, c_out))
    model __
    
    
    TCN(
      (tcn): Sequential(
        (0): TemporalBlock(
          (conv1): Conv1d(3, 25, kernel_size=(7,), stride=(1,), padding=(6,))
          (chomp1): Chomp1d()
          (relu1): ReLU()
          (dropout1): Dropout(p=0.0, inplace=False)
          (conv2): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(6,))
          (chomp2): Chomp1d()
          (relu2): ReLU()
          (dropout2): Dropout(p=0.0, inplace=False)
          (net): Sequential(
            (0): Conv1d(3, 25, kernel_size=(7,), stride=(1,), padding=(6,))
            (1): Chomp1d()
            (2): ReLU()
            (3): Dropout(p=0.0, inplace=False)
            (4): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(6,))
            (5): Chomp1d()
            (6): ReLU()
            (7): Dropout(p=0.0, inplace=False)
          )
          (downsample): Conv1d(3, 25, kernel_size=(1,), stride=(1,))
          (relu): ReLU()
        )
        (1): TemporalBlock(
          (conv1): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(12,), dilation=(2,))
          (chomp1): Chomp1d()
          (relu1): ReLU()
          (dropout1): Dropout(p=0.0, inplace=False)
          (conv2): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(12,), dilation=(2,))
          (chomp2): Chomp1d()
          (relu2): ReLU()
          (dropout2): Dropout(p=0.0, inplace=False)
          (net): Sequential(
            (0): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(12,), dilation=(2,))
            (1): Chomp1d()
            (2): ReLU()
            (3): Dropout(p=0.0, inplace=False)
            (4): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(12,), dilation=(2,))
            (5): Chomp1d()
            (6): ReLU()
            (7): Dropout(p=0.0, inplace=False)
          )
          (relu): ReLU()
        )
        (2): TemporalBlock(
          (conv1): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(24,), dilation=(4,))
          (chomp1): Chomp1d()
          (relu1): ReLU()
          (dropout1): Dropout(p=0.0, inplace=False)
          (conv2): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(24,), dilation=(4,))
          (chomp2): Chomp1d()
          (relu2): ReLU()
          (dropout2): Dropout(p=0.0, inplace=False)
          (net): Sequential(
            (0): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(24,), dilation=(4,))
            (1): Chomp1d()
            (2): ReLU()
            (3): Dropout(p=0.0, inplace=False)
            (4): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(24,), dilation=(4,))
            (5): Chomp1d()
            (6): ReLU()
            (7): Dropout(p=0.0, inplace=False)
          )
          (relu): ReLU()
        )
        (3): TemporalBlock(
          (conv1): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(48,), dilation=(8,))
          (chomp1): Chomp1d()
          (relu1): ReLU()
          (dropout1): Dropout(p=0.0, inplace=False)
          (conv2): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(48,), dilation=(8,))
          (chomp2): Chomp1d()
          (relu2): ReLU()
          (dropout2): Dropout(p=0.0, inplace=False)
          (net): Sequential(
            (0): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(48,), dilation=(8,))
            (1): Chomp1d()
            (2): ReLU()
            (3): Dropout(p=0.0, inplace=False)
            (4): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(48,), dilation=(8,))
            (5): Chomp1d()
            (6): ReLU()
            (7): Dropout(p=0.0, inplace=False)
          )
          (relu): ReLU()
        )
        (4): TemporalBlock(
          (conv1): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(96,), dilation=(16,))
          (chomp1): Chomp1d()
          (relu1): ReLU()
          (dropout1): Dropout(p=0.0, inplace=False)
          (conv2): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(96,), dilation=(16,))
          (chomp2): Chomp1d()
          (relu2): ReLU()
          (dropout2): Dropout(p=0.0, inplace=False)
          (net): Sequential(
            (0): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(96,), dilation=(16,))
            (1): Chomp1d()
            (2): ReLU()
            (3): Dropout(p=0.0, inplace=False)
            (4): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(96,), dilation=(16,))
            (5): Chomp1d()
            (6): ReLU()
            (7): Dropout(p=0.0, inplace=False)
          )
          (relu): ReLU()
        )
        (5): TemporalBlock(
          (conv1): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(192,), dilation=(32,))
          (chomp1): Chomp1d()
          (relu1): ReLU()
          (dropout1): Dropout(p=0.0, inplace=False)
          (conv2): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(192,), dilation=(32,))
          (chomp2): Chomp1d()
          (relu2): ReLU()
          (dropout2): Dropout(p=0.0, inplace=False)
          (net): Sequential(
            (0): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(192,), dilation=(32,))
            (1): Chomp1d()
            (2): ReLU()
            (3): Dropout(p=0.0, inplace=False)
            (4): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(192,), dilation=(32,))
            (5): Chomp1d()
            (6): ReLU()
            (7): Dropout(p=0.0, inplace=False)
          )
          (relu): ReLU()
        )
        (6): TemporalBlock(
          (conv1): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(384,), dilation=(64,))
          (chomp1): Chomp1d()
          (relu1): ReLU()
          (dropout1): Dropout(p=0.0, inplace=False)
          (conv2): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(384,), dilation=(64,))
          (chomp2): Chomp1d()
          (relu2): ReLU()
          (dropout2): Dropout(p=0.0, inplace=False)
          (net): Sequential(
            (0): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(384,), dilation=(64,))
            (1): Chomp1d()
            (2): ReLU()
            (3): Dropout(p=0.0, inplace=False)
            (4): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(384,), dilation=(64,))
            (5): Chomp1d()
            (6): ReLU()
            (7): Dropout(p=0.0, inplace=False)
          )
          (relu): ReLU()
        )
        (7): TemporalBlock(
          (conv1): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(768,), dilation=(128,))
          (chomp1): Chomp1d()
          (relu1): ReLU()
          (dropout1): Dropout(p=0.0, inplace=False)
          (conv2): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(768,), dilation=(128,))
          (chomp2): Chomp1d()
          (relu2): ReLU()
          (dropout2): Dropout(p=0.0, inplace=False)
          (net): Sequential(
            (0): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(768,), dilation=(128,))
            (1): Chomp1d()
            (2): ReLU()
            (3): Dropout(p=0.0, inplace=False)
            (4): Conv1d(25, 25, kernel_size=(7,), stride=(1,), padding=(768,), dilation=(128,))
            (5): Chomp1d()
            (6): ReLU()
            (7): Dropout(p=0.0, inplace=False)
          )
          (relu): ReLU()
        )
      )
      (gap): GAP1d(
        (gap): AdaptiveAvgPool1d(output_size=1)
        (flatten): Flatten(full=False)
      )
      (linear): Linear(in_features=25, out_features=2, bias=True)
    )

  * __Report an issue


