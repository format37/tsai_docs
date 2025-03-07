## On this page

  * TSIdentity
  * TSShuffle_HLs
  * TSShuffleSteps
  * TSGaussianNoise
  * TSMagMulNoise
  * TSMagAddNoise
  * random_cum_linear_generator
  * random_cum_noise_generator
  * random_cum_curve_generator
  * random_curve_generator
  * TSTimeNoise
  * TSMagWarp
  * TSTimeWarp
  * TSWindowWarp
  * TSMagScalePerVar
  * TSMagScale
  * test_interpolate
  * TSRandomResizedCrop
  * TSWindowSlicing
  * TSRandomZoomOut
  * TSRandomTimeScale
  * TSRandomTimeStep
  * TSResampleSteps
  * TSBlur
  * TSSmooth
  * TSFreqDenoise
  * maddest
  * TSRandomFreqNoise
  * TSRandomResizedLookBack
  * TSRandomLookBackOut
  * TSVarOut
  * TSCutOut
  * TSTimeStepOut
  * TSRandomCropPad
  * TSMaskOut
  * TSInputDropout
  * TSTranslateX
  * TSRandomShift
  * TSHorizontalFlip
  * TSRandomTrend
  * TSVerticalFlip
  * TSResize
  * TSRandomSize
  * TSRandomLowRes
  * TSDownUpScale
  * TSRandomDownUpScale
  * TSRandomConv
  * TSRandom2Value
  * TSMask2Value
  * TSSelfDropout
  * self_mask
  * RandAugment
  * TestTfm
  * get_tfm_name



  * __Report an issue



  1. Data
  2. Time Series Data Augmentation



# Time Series Data Augmentation

> Functions used to transform TSTensors (Data Augmentation)
    
    
    from tsai.data.core import TSCategorize
    from tsai.data.external import get_UCR_data
    from tsai.data.preprocessing import TSStandardize __
    
    
    dsid = 'NATOPS'
    X, y, splits = get_UCR_data(dsid, return_split=False)
    tfms = [None, TSCategorize()]
    batch_tfms = TSStandardize()
    dls = get_ts_dls(X, y, tfms=tfms, splits=splits, batch_tfms=batch_tfms, bs=128)
    xb, yb = next(iter(dls.train))__

* * *

source

### TSIdentity

> 
>      TSIdentity (magnitude=None, **kwargs)

_Applies the identity tfm to a`TSTensor` batch_
    
    
    test_eq(TSIdentity()(xb, split_idx=0).shape, xb.shape)__

* * *

source

### TSShuffle_HLs

> 
>      TSShuffle_HLs (magnitude=1.0, ex=None, **kwargs)

_Randomly shuffles HIs/LOs of an OHLC`TSTensor` batch_
    
    
    test_eq(TSShuffle_HLs()(xb, split_idx=0).shape, xb.shape)__

* * *

source

### TSShuffleSteps

> 
>      TSShuffleSteps (magnitude=1.0, ex=None, **kwargs)

_Randomly shuffles consecutive sequence datapoints in batch_
    
    
    t = TSTensor(torch.arange(11).float())
    tt_ = []
    for _ in range(1000):
        tt = TSShuffleSteps()(t, split_idx=0)
        test_eq(len(set(tt.tolist())), len(t))
        test_ne(tt, t)
        tt_.extend([t for i,t in enumerate(tt) if t!=i])
    x, y = np.unique(tt_, return_counts=True) # This is to visualize distribution which should be equal for all and half for first and last items
    plt.bar(x, y);__

* * *

source

### TSGaussianNoise

> 
>      TSGaussianNoise (magnitude=0.5, additive=True, ex=None, **kwargs)

_Applies additive or multiplicative gaussian noise_
    
    
    test_eq(TSGaussianNoise(.1, additive=True)(xb, split_idx=0).shape, xb.shape)
    test_eq(TSGaussianNoise(.1, additive=False)(xb, split_idx=0).shape, xb.shape)__

* * *

source

### TSMagMulNoise

> 
>      TSMagMulNoise (magnitude=1, ex=None, **kwargs)

_Applies multiplicative noise on the y-axis for each step of a`TSTensor` batch_

* * *

source

### TSMagAddNoise

> 
>      TSMagAddNoise (magnitude=1, ex=None, **kwargs)

_Applies additive noise on the y-axis for each step of a`TSTensor` batch_
    
    
    test_eq(TSMagAddNoise()(xb, split_idx=0).shape, xb.shape)
    test_eq(TSMagMulNoise()(xb, split_idx=0).shape, xb.shape)
    test_ne(TSMagAddNoise()(xb, split_idx=0), xb)
    test_ne(TSMagMulNoise()(xb, split_idx=0), xb)__

* * *

source

### random_cum_linear_generator

> 
>      random_cum_linear_generator (o, magnitude=0.1)

* * *

source

### random_cum_noise_generator

> 
>      random_cum_noise_generator (o, magnitude=0.1, noise=None)

* * *

source

### random_cum_curve_generator

> 
>      random_cum_curve_generator (o, magnitude=0.1, order=4, noise=None)

* * *

source

### random_curve_generator

> 
>      random_curve_generator (o, magnitude=0.1, order=4, noise=None)

* * *

source

### TSTimeNoise

> 
>      TSTimeNoise (magnitude=0.1, ex=None, **kwargs)

_Applies noise to each step in the x-axis of a`TSTensor` batch based on smooth random curve_
    
    
    test_eq(TSTimeNoise()(xb, split_idx=0).shape, xb.shape)
    test_ne(TSTimeNoise()(xb, split_idx=0), xb)__

* * *

source

### TSMagWarp

> 
>      TSMagWarp (magnitude=0.02, ord=4, ex=None, **kwargs)

_Applies warping to the y-axis of a`TSTensor` batch based on a smooth random curve_
    
    
    test_eq(TSMagWarp()(xb, split_idx=0).shape, xb.shape)
    test_ne(TSMagWarp()(xb, split_idx=0), xb)__

* * *

source

### TSTimeWarp

> 
>      TSTimeWarp (magnitude=0.1, ord=6, ex=None, **kwargs)

_Applies time warping to the x-axis of a`TSTensor` batch based on a smooth random curve_
    
    
    test_eq(TSTimeWarp()(xb, split_idx=0).shape, xb.shape)
    test_ne(TSTimeWarp()(xb, split_idx=0), xb)__

* * *

source

### TSWindowWarp

> 
>      TSWindowWarp (magnitude=0.1, ex=None, **kwargs)

_Applies window slicing to the x-axis of a`TSTensor` batch based on a random linear curve based on https://halshs.archives-ouvertes.fr/halshs-01357973/document_
    
    
    test_eq(TSWindowWarp()(xb, split_idx=0).shape, xb.shape)__

* * *

source

### TSMagScalePerVar

> 
>      TSMagScalePerVar (magnitude=0.5, ex=None, **kwargs)

_Applies per_var scaling to the y-axis of a`TSTensor` batch based on a scalar_

* * *

source

### TSMagScale

> 
>      TSMagScale (magnitude=0.5, ex=None, **kwargs)

_Applies scaling to the y-axis of a`TSTensor` batch based on a scalar_
    
    
    test_eq(TSMagScale()(xb, split_idx=0).shape, xb.shape)
    test_eq(TSMagScalePerVar()(xb, split_idx=0).shape, xb.shape)
    test_ne(TSMagScale()(xb, split_idx=0), xb)
    test_ne(TSMagScalePerVar()(xb, split_idx=0), xb)__

* * *

source

### test_interpolate

> 
>      test_interpolate (mode='linear')
    
    
    # Run the test
    test_interpolate('linear')__
    
    
    linear interpolation is not supported by mps. You can try a different mode
    Error: The operator 'aten::upsample_linear1d.out' is not currently implemented for the MPS device. If you want this op to be added in priority during the prototype phase of this feature, please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.
    
    
    False
    
    
    test_interpolate('nearest')__
    
    
    True

* * *

source

### TSRandomResizedCrop

> 
>      TSRandomResizedCrop (magnitude=0.1, size=None, scale=None, ex=None,
>                           mode='nearest', **kwargs)

_Randomly amplifies a sequence focusing on a random section of the steps_
    
    
    if test_interpolate('nearest'):
        test_eq(TSRandomResizedCrop(.5)(xb, split_idx=0).shape, xb.shape)
        test_ne(TSRandomResizedCrop(size=.8, scale=(.5, 1))(xb, split_idx=0).shape, xb.shape)
        test_ne(TSRandomResizedCrop(size=20, scale=(.5, 1))(xb, split_idx=0).shape, xb.shape)__

* * *

source

### TSWindowSlicing

> 
>      TSWindowSlicing (magnitude=0.1, ex=None, mode='nearest', **kwargs)

_Randomly extracts an resize a ts slice based on https://halshs.archives-ouvertes.fr/halshs-01357973/document_
    
    
    if test_interpolate('nearest'):
        test_eq(TSWindowSlicing()(xb, split_idx=0).shape, xb.shape)
        test_ne(TSWindowSlicing()(xb, split_idx=0), xb)__

* * *

source

### TSRandomZoomOut

> 
>      TSRandomZoomOut (magnitude=0.1, ex=None, mode='nearest', **kwargs)

_Randomly compresses a sequence on the x-axis_
    
    
    if test_interpolate('nearest'):
        test_eq(TSRandomZoomOut(.5)(xb, split_idx=0).shape, xb.shape)#__

* * *

source

### TSRandomTimeScale

> 
>      TSRandomTimeScale (magnitude=0.1, ex=None, mode='nearest', **kwargs)

_Randomly amplifies/ compresses a sequence on the x-axis keeping the same length_
    
    
    if test_interpolate('nearest'):
        test_eq(TSRandomTimeScale(.5)(xb, split_idx=0).shape, xb.shape)__

* * *

source

### TSRandomTimeStep

> 
>      TSRandomTimeStep (magnitude=0.02, ex=None, mode='nearest', **kwargs)

_Compresses a sequence on the x-axis by randomly selecting sequence steps and interpolating to previous size_
    
    
    if test_interpolate('nearest'):
        test_eq(TSRandomTimeStep()(xb, split_idx=0).shape, xb.shape)__

* * *

source

### TSResampleSteps

> 
>      TSResampleSteps (step_pct=1.0, same_seq_len=True, magnitude=None,
>                       **kwargs)

_Transform that randomly selects and sorts sequence steps (with replacement) maintaining the sequence length_
    
    
    test_eq(TSResampleSteps(step_pct=.9, same_seq_len=False)(xb, split_idx=0).shape[-1], round(.9*xb.shape[-1]))
    test_eq(TSResampleSteps(step_pct=.9, same_seq_len=True)(xb, split_idx=0).shape[-1], xb.shape[-1])__

* * *

source

### TSBlur

> 
>      TSBlur (magnitude=1.0, ex=None, filt_len=None, **kwargs)

_Blurs a sequence applying a filter of type [1, 0, 1]_
    
    
    test_eq(TSBlur(filt_len=7)(xb, split_idx=0).shape, xb.shape)
    test_ne(TSBlur()(xb, split_idx=0), xb)__

* * *

source

### TSSmooth

> 
>      TSSmooth (magnitude=1.0, ex=None, filt_len=None, **kwargs)

_Smoothens a sequence applying a filter of type [1, 5, 1]_
    
    
    test_eq(TSSmooth(filt_len=7)(xb, split_idx=0).shape, xb.shape)
    test_ne(TSSmooth()(xb, split_idx=0), xb)__

* * *

source

### TSFreqDenoise

> 
>      TSFreqDenoise (magnitude=0.1, ex=None, wavelet='db4', level=2, thr=None,
>                     thr_mode='hard', pad_mode='per', **kwargs)

_Denoises a sequence applying a wavelet decomposition method_

* * *

source

### maddest

> 
>      maddest (d, axis=None)
    
    
    try: import pywt
    except ImportError: pass __
    
    
    if 'pywt' in dir():
        test_eq(TSFreqDenoise()(xb, split_idx=0).shape, xb.shape)
        test_ne(TSFreqDenoise()(xb, split_idx=0), xb)__

* * *

source

### TSRandomFreqNoise

> 
>      TSRandomFreqNoise (magnitude=0.1, ex=None, wavelet='db4', level=2,
>                         mode='constant', **kwargs)

_Applys random noise using a wavelet decomposition method_
    
    
    if 'pywt' in dir():
        test_eq(TSRandomFreqNoise()(xb, split_idx=0).shape, xb.shape)__

* * *

source

### TSRandomResizedLookBack

> 
>      TSRandomResizedLookBack (magnitude=0.1, mode='nearest', **kwargs)

_Selects a random number of sequence steps starting from the end and return an output of the same shape_
    
    
    if test_interpolate('nearest'):
        for i in range(100):
            o = TSRandomResizedLookBack()(xb, split_idx=0)
            test_eq(o.shape[-1], xb.shape[-1])__

* * *

source

### TSRandomLookBackOut

> 
>      TSRandomLookBackOut (magnitude=0.1, **kwargs)

_Selects a random number of sequence steps starting from the end and set them to zero_
    
    
    for i in range(100):
        o = TSRandomLookBackOut()(xb, split_idx=0)
        test_eq(o.shape[-1], xb.shape[-1])__

* * *

source

### TSVarOut

> 
>      TSVarOut (magnitude=0.05, ex=None, **kwargs)

_Set the value of a random number of variables to zero_
    
    
    test_eq(TSVarOut()(xb, split_idx=0).shape, xb.shape)__

* * *

source

### TSCutOut

> 
>      TSCutOut (magnitude=0.05, ex=None, **kwargs)

_Sets a random section of the sequence to zero_
    
    
    test_eq(TSCutOut()(xb, split_idx=0).shape, xb.shape)__

* * *

source

### TSTimeStepOut

> 
>      TSTimeStepOut (magnitude=0.05, ex=None, **kwargs)

_Sets random sequence steps to zero_
    
    
    test_eq(TSTimeStepOut()(xb, split_idx=0).shape, xb.shape)__

* * *

source

### TSRandomCropPad

> 
>      TSRandomCropPad (magnitude=0.05, ex=None, **kwargs)

_Crops a section of the sequence of a random length_
    
    
    test_eq(TSRandomCropPad()(xb, split_idx=0).shape, xb.shape)__

* * *

source

### TSMaskOut

> 
>      TSMaskOut (magnitude=0.1, compensate:bool=False, ex=None, **kwargs)

_Applies a random mask_
    
    
    test_eq(TSMaskOut()(xb, split_idx=0).shape, xb.shape)
    test_ne(TSMaskOut()(xb, split_idx=0), xb)__

* * *

source

### TSInputDropout

> 
>      TSInputDropout (magnitude=0.0, ex=None, **kwargs)

_Applies input dropout with required_grad=False_
    
    
    test_eq(TSInputDropout(.1)(xb, split_idx=0).shape, xb.shape)
    test_ne(TSInputDropout(.1)(xb, split_idx=0), xb)__

* * *

source

### TSTranslateX

> 
>      TSTranslateX (magnitude=0.1, ex=None, **kwargs)

_Moves a selected sequence window a random number of steps_
    
    
    test_eq(TSTranslateX()(xb, split_idx=0).shape, xb.shape)__

* * *

source

### TSRandomShift

> 
>      TSRandomShift (magnitude=0.02, ex=None, **kwargs)

_Shifts and splits a sequence_
    
    
    test_eq(TSRandomShift()(xb, split_idx=0).shape, xb.shape)__

* * *

source

### TSHorizontalFlip

> 
>      TSHorizontalFlip (magnitude=1.0, ex=None, **kwargs)

_Flips the sequence along the x-axis_
    
    
    test_eq(TSHorizontalFlip()(xb, split_idx=0).shape, xb.shape)
    test_ne(TSHorizontalFlip()(xb, split_idx=0), xb)__

* * *

source

### TSRandomTrend

> 
>      TSRandomTrend (magnitude=0.1, ex=None, **kwargs)

_Randomly rotates the sequence along the z-axis_
    
    
    test_eq(TSRandomTrend()(xb, split_idx=0).shape, xb.shape)__

* * *

source

### TSVerticalFlip

> 
>      TSVerticalFlip (magnitude=1.0, ex=None, **kwargs)

_Applies a negative value to the time sequence_
    
    
    test_eq(TSVerticalFlip()(xb, split_idx=0).shape, xb.shape)
    test_ne(TSVerticalFlip()(xb, split_idx=0), xb)__

* * *

source

### TSResize

> 
>      TSResize (magnitude=-0.5, size=None, ex=None, mode='nearest', **kwargs)

_Resizes the sequence length of a time series_
    
    
    if test_interpolate('nearest'):
        for sz in np.linspace(.2, 2, 10): test_eq(TSResize(sz)(xb, split_idx=0).shape[-1], int(round(xb.shape[-1]*(1+sz))))
        test_ne(TSResize(1)(xb, split_idx=0), xb)__

* * *

source

### TSRandomSize

> 
>      TSRandomSize (magnitude=0.1, ex=None, mode='nearest', **kwargs)

_Randomly resizes the sequence length of a time series_
    
    
    if test_interpolate('nearest'):
        seq_len_ = []
        for i in range(100):
            o = TSRandomSize(.5)(xb, split_idx=0)
            seq_len_.append(o.shape[-1])
        test_lt(min(seq_len_), xb.shape[-1])
        test_gt(max(seq_len_), xb.shape[-1])__

* * *

source

### TSRandomLowRes

> 
>      TSRandomLowRes (magnitude=0.5, ex=None, mode='nearest', **kwargs)

_Randomly resizes the sequence length of a time series to a lower resolution_

* * *

source

### TSDownUpScale

> 
>      TSDownUpScale (magnitude=0.5, ex=None, mode='nearest', **kwargs)

_Downscales a time series and upscales it again to previous sequence length_
    
    
    if test_interpolate('nearest'):
        test_eq(TSDownUpScale()(xb, split_idx=0).shape, xb.shape)__

* * *

source

### TSRandomDownUpScale

> 
>      TSRandomDownUpScale (magnitude=0.5, ex=None, mode='nearest', **kwargs)

_Randomly downscales a time series and upscales it again to previous sequence length_
    
    
    if test_interpolate('nearest'):
        test_eq(TSRandomDownUpScale()(xb, split_idx=0).shape, xb.shape)
        test_ne(TSDownUpScale()(xb, split_idx=0), xb)
        test_eq(TSDownUpScale()(xb, split_idx=1), xb)__

* * *

source

### TSRandomConv

> 
>      TSRandomConv (magnitude=0.05, ex=None, ks=[1, 3, 5, 7], **kwargs)

_Applies a convolution with a random kernel and random weights with required_grad=False_
    
    
    for i in range(5):
        o = TSRandomConv(magnitude=0.05, ex=None, ks=[1, 3, 5, 7])(xb, split_idx=0)
        test_eq(o.shape, xb.shape)__

* * *

source

### TSRandom2Value

> 
>      TSRandom2Value (magnitude=0.1, sel_vars=None, sel_steps=None,
>                      static=False, value=nan, **kwargs)

_Randomly sets selected variables of type`TSTensor` to predefined value (default: np.nan)_
    
    
    t = TSTensor(torch.ones(2, 3, 10))
    TSRandom2Value(magnitude=0.5, sel_vars=None, sel_steps=None, static=False, value=0)(t, split_idx=0).data __
    
    
    tensor([[[0., 0., 1., 0., 1., 1., 0., 1., 1., 0.],
             [1., 1., 0., 1., 1., 1., 1., 1., 1., 0.],
             [1., 1., 1., 1., 1., 0., 0., 1., 1., 1.]],
    
            [[1., 1., 1., 1., 1., 0., 1., 1., 0., 1.],
             [0., 0., 0., 0., 0., 1., 0., 1., 0., 1.],
             [0., 1., 0., 1., 0., 0., 0., 1., 0., 0.]]])
    
    
    t = TSTensor(torch.ones(2, 3, 10))
    TSRandom2Value(magnitude=0.5, sel_vars=[1], sel_steps=slice(-5, None), static=False, value=0)(t, split_idx=0).data __
    
    
    tensor([[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 0., 1., 0., 0., 0.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]],
    
            [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 0., 1., 0., 0., 0.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]])
    
    
    t = TSTensor(torch.ones(2, 3, 10))
    TSRandom2Value(magnitude=0.5, sel_vars=[1], sel_steps=None, static=True, value=0)(t, split_idx=0).data __
    
    
    tensor([[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]],
    
            [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]])
    
    
    t = TSTensor(torch.ones(2, 3, 10))
    TSRandom2Value(magnitude=1, sel_vars=1, sel_steps=None, static=False, value=0)(t, split_idx=0).data __
    
    
    tensor([[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]],
    
            [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]])
    
    
    t = TSTensor(torch.ones(2, 3, 10))
    TSRandom2Value(magnitude=1, sel_vars=[1,2], sel_steps=None, static=False, value=0)(t, split_idx=0).data __
    
    
    tensor([[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
    
            [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])
    
    
    t = TSTensor(torch.ones(2, 3, 10))
    TSRandom2Value(magnitude=1, sel_vars=1, sel_steps=[1,3,5], static=False, value=0)(t, split_idx=0).data __
    
    
    tensor([[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 0., 1., 0., 1., 0., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]],
    
            [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 0., 1., 0., 1., 0., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]])
    
    
    t = TSTensor(torch.ones(2, 3, 10))
    TSRandom2Value(magnitude=1, sel_vars=[1,2], sel_steps=[1,3,5], static=False, value=0)(t, split_idx=0).data __
    
    
    tensor([[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 0., 1., 0., 1., 0., 1., 1., 1., 1.],
             [1., 0., 1., 0., 1., 0., 1., 1., 1., 1.]],
    
            [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 0., 1., 0., 1., 0., 1., 1., 1., 1.],
             [1., 0., 1., 0., 1., 0., 1., 1., 1., 1.]]])
    
    
    t = TSTensor(torch.ones(2,3,4))
    TSRandom2Value(magnitude=.5, sel_vars=[0,2])(t, split_idx=0).data __
    
    
    tensor([[[1., nan, nan, 1.],
             [1., 1., 1., 1.],
             [1., nan, 1., 1.]],
    
            [[nan, 1., 1., nan],
             [1., 1., 1., 1.],
             [nan, nan, 1., 1.]]])
    
    
    t = TSTensor(torch.ones(2,3,4))
    TSRandom2Value(magnitude=.5, sel_steps=slice(2, None))(t, split_idx=0).data __
    
    
    tensor([[[1., 1., 1., nan],
             [1., 1., nan, 1.],
             [1., 1., nan, nan]],
    
            [[1., 1., nan, 1.],
             [1., 1., nan, nan],
             [1., 1., nan, 1.]]])
    
    
    t = TSTensor(torch.ones(2,3,100))
    test_gt(np.isnan(TSRandom2Value(magnitude=.5)(t, split_idx=0)).sum().item(), 0)
    t = TSTensor(torch.ones(2,3,100))
    test_gt(np.isnan(TSRandom2Value(magnitude=.5, sel_vars=[0,2])(t, split_idx=0)[:, [0,2]]).sum().item(), 0)
    t = TSTensor(torch.ones(2,3,100))
    test_eq(np.isnan(TSRandom2Value(magnitude=.5, sel_vars=[0,2])(t, split_idx=0)[:, 1]).sum().item(), 0)__

* * *

source

### TSMask2Value

> 
>      TSMask2Value (mask_fn, value=nan, sel_vars=None, **kwargs)

_Randomly sets selected variables of type`TSTensor` to predefined value (default: np.nan)_
    
    
    t = TSTensor(torch.ones(2,3,100))
    def _mask_fn(o, r=.15, value=np.nan):
        return torch.rand_like(o) > (1-r)
    test_gt(np.isnan(TSMask2Value(_mask_fn)(t, split_idx=0)).sum().item(), 0)__

* * *

source

### TSSelfDropout

> 
>      TSSelfDropout (p:float=1.0, nm:str=None, before_call:callable=None,
>                     **kwargs)

_Applies dropout to a tensor with nan values by rotating axis=0 inplace_

| **Type** | **Default** | **Details**  
---|---|---|---  
p | float | 1.0 | Probability of applying Transform  
nm | str | None |   
before_call | callable | None | Optional batchwise preprocessing function  
kwargs | VAR_KEYWORD |  |   
  
* * *

source

### self_mask

> 
>      self_mask (o)
    
    
    t = TSTensor(torch.ones(2,3,100))
    mask = torch.rand_like(t) > .7
    t[mask] = np.nan
    nan_perc = np.isnan(t).float().mean().item()
    t2 = TSSelfDropout()(t, split_idx=0)
    test_gt(torch.isnan(t2).float().mean().item(), nan_perc)
    nan_perc, torch.isnan(t2).float().mean().item()__
    
    
    (0.30000001192092896, 0.49000000953674316)

* * *

source

### RandAugment

> 
>      RandAugment (tfms:list, N:int=1, M:int=3, **kwargs)

_A transform that before_call its state at each`__call__`_
    
    
    test_ne(RandAugment(TSMagAddNoise, N=5, M=10)(xb, split_idx=0), xb)__

* * *

source

### TestTfm

> 
>      TestTfm (tfm, magnitude=1.0, ex=None, **kwargs)

_Utility class to test the output of selected tfms during training_

* * *

source

### get_tfm_name

> 
>      get_tfm_name (tfm)
    
    
    test_eq(get_tfm_name(partial(TSMagScale()))==get_tfm_name((partial(TSMagScale()), 0.1, .05))==get_tfm_name(TSMagScale())==get_tfm_name((TSMagScale(), 0.1, .05)), True)__
    
    
    all_TS_randaugs_names = [get_tfm_name(t) for t in all_TS_randaugs]__

  * __Report an issue


