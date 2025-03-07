## On this page

  * GRUPlus
  * LSTMPlus
  * RNNPlus
  * Converting a model to TorchScript
    * Tracing
  * Converting a model to ONNX



  * __Report an issue



  1. Models
  2. RNNs
  3. RNNPlus



# RNNPlus

These are RNN, LSTM and GRU PyTorch implementations created by Ignacio Oguiza - oguiza@timeseriesAI.co

The idea of including a feature extractor to the RNN network comes from the solution developed by the UPSTAGE team (https://www.kaggle.com/songwonho, https://www.kaggle.com/limerobot and https://www.kaggle.com/jungikhyo). They finished in 3rd position in Kaggle’s Google Brain - Ventilator Pressure Prediction competition. They used a Conv1d + Stacked LSTM architecture.

* * *

source

### GRUPlus

> 
>      GRUPlus (c_in, c_out, seq_len=None, hidden_size=[100], n_layers=1,
>               bias=True, rnn_dropout=0, bidirectional=False,
>               n_cat_embeds=None, cat_embed_dims=None, cat_padding_idxs=None,
>               cat_pos=None, feature_extractor=None, fc_dropout=0.0,
>               last_step=True, bn=False, custom_head=None, y_range=None,
>               init_weights=True, **kwargs)

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

### LSTMPlus

> 
>      LSTMPlus (c_in, c_out, seq_len=None, hidden_size=[100], n_layers=1,
>                bias=True, rnn_dropout=0, bidirectional=False,
>                n_cat_embeds=None, cat_embed_dims=None, cat_padding_idxs=None,
>                cat_pos=None, feature_extractor=None, fc_dropout=0.0,
>                last_step=True, bn=False, custom_head=None, y_range=None,
>                init_weights=True, **kwargs)

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

### RNNPlus

> 
>      RNNPlus (c_in, c_out, seq_len=None, hidden_size=[100], n_layers=1,
>               bias=True, rnn_dropout=0, bidirectional=False,
>               n_cat_embeds=None, cat_embed_dims=None, cat_padding_idxs=None,
>               cat_pos=None, feature_extractor=None, fc_dropout=0.0,
>               last_step=True, bn=False, custom_head=None, y_range=None,
>               init_weights=True, **kwargs)

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
    
    
    bs = 16
    c_in = 3
    seq_len = 12
    c_out = 2
    xb = torch.rand(bs, c_in, seq_len)
    test_eq(RNNPlus(c_in, c_out)(xb).shape, [bs, c_out])
    test_eq(RNNPlus(c_in, c_out, hidden_size=100, n_layers=2, bias=True, rnn_dropout=0.2, bidirectional=True, fc_dropout=0.5)(xb).shape, 
            [bs, c_out])
    test_eq(RNNPlus(c_in, c_out, hidden_size=[100, 50, 10], bias=True, rnn_dropout=0.2, bidirectional=True, fc_dropout=0.5)(xb).shape, 
            [bs, c_out])
    test_eq(RNNPlus(c_in, c_out, hidden_size=[100], n_layers=2, bias=True, rnn_dropout=0.2, bidirectional=True, fc_dropout=0.5)(xb).shape, 
            [bs, c_out])
    test_eq(LSTMPlus(c_in, c_out, hidden_size=100, n_layers=2, bias=True, rnn_dropout=0.2, bidirectional=True, fc_dropout=0.5)(xb).shape, 
            [bs, c_out])
    test_eq(GRUPlus(c_in, c_out, hidden_size=100, n_layers=2, bias=True, rnn_dropout=0.2, bidirectional=True, fc_dropout=0.5)(xb).shape, 
            [bs, c_out])
    test_eq(RNNPlus(c_in, c_out, seq_len, last_step=False)(xb).shape, [bs, c_out])
    test_eq(RNNPlus(c_in, c_out, seq_len, last_step=False)(xb).shape, [bs, c_out])
    test_eq(RNNPlus(c_in, c_out, seq_len, hidden_size=100, n_layers=2, bias=True, rnn_dropout=0.2, bidirectional=True, fc_dropout=0.5, 
                    last_step=False)(xb).shape, 
            [bs, c_out])
    test_eq(LSTMPlus(c_in, c_out, seq_len, last_step=False)(xb).shape, [bs, c_out])
    test_eq(GRUPlus(c_in, c_out, seq_len, last_step=False)(xb).shape, [bs, c_out])__
    
    
    feature_extractor = MultiConv1d(c_in, kss=[1,3,5,7])
    custom_head = nn.Sequential(Transpose(1,2), nn.Linear(8,8), nn.SELU(), nn.Linear(8, 1), Squeeze())
    test_eq(LSTMPlus(c_in, c_out, seq_len, hidden_size=[32,16,8,4], bidirectional=True, 
                     feature_extractor=feature_extractor, custom_head=custom_head)(xb).shape, [bs, seq_len])
    feature_extractor = MultiConv1d(c_in, kss=[1,3,5,7], keep_original=True)
    custom_head = nn.Sequential(Transpose(1,2), nn.Linear(8,8), nn.SELU(), nn.Linear(8, 1), Squeeze())
    test_eq(LSTMPlus(c_in, c_out, seq_len, hidden_size=[32,16,8,4], bidirectional=True, 
                     feature_extractor=feature_extractor, custom_head=custom_head)(xb).shape, [bs, seq_len])__
    
    
    [W NNPACK.cpp:53] Could not initialize NNPACK! Reason: Unsupported hardware.
    
    
    bs = 16
    c_in = 3
    seq_len = 12
    c_out = 2
    x1 = torch.rand(bs,1,seq_len)
    x2 = torch.randint(0,3,(bs,1,seq_len))
    x3 = torch.randint(0,5,(bs,1,seq_len))
    xb = torch.cat([x1,x2,x3],1)
    
    custom_head = partial(create_mlp_head, fc_dropout=0.5)
    test_eq(LSTMPlus(c_in, c_out, seq_len, last_step=False, custom_head=custom_head)(xb).shape, [bs, c_out])
    custom_head = partial(create_pool_head, concat_pool=True, fc_dropout=0.5)
    test_eq(LSTMPlus(c_in, c_out, seq_len, last_step=False, custom_head=custom_head)(xb).shape, [bs, c_out])
    custom_head = partial(create_pool_plus_head, fc_dropout=0.5)
    test_eq(LSTMPlus(c_in, c_out, seq_len, last_step=False, custom_head=custom_head)(xb).shape, [bs, c_out])
    custom_head = partial(create_conv_head)
    test_eq(LSTMPlus(c_in, c_out, seq_len, last_step=False, custom_head=custom_head)(xb).shape, [bs, c_out])
    test_eq(LSTMPlus(c_in, c_out, seq_len, hidden_size=[100, 50], n_layers=2, bias=True, rnn_dropout=0.2, bidirectional=True)(xb).shape, 
            [bs, c_out])
    
    n_cat_embeds = [3, 5]
    cat_pos = [1, 2]
    custom_head = partial(create_conv_head)
    m = LSTMPlus(c_in, c_out, seq_len, hidden_size=[100, 50], n_layers=2, bias=True, rnn_dropout=0.2, bidirectional=True, 
                 n_cat_embeds=n_cat_embeds, cat_pos=cat_pos)
    test_eq(m(xb).shape, [bs, c_out])__
    
    
    from tsai.data.all import *
    from tsai.models.utils import *__
    
    
    dsid = 'NATOPS' 
    bs = 16
    X, y, splits = get_UCR_data(dsid, return_split=False)
    tfms  = [None, [Categorize()]]
    dls = get_ts_dls(X, y, tfms=tfms, splits=splits, bs=bs)__
    
    
    model = build_ts_model(LSTMPlus, dls=dls)
    print(model[-1])
    learn = Learner(dls, model,  metrics=accuracy)
    learn.fit_one_cycle(1, 3e-3)__
    
    
    Sequential(
      (0): LastStep()
      (1): Linear(in_features=100, out_features=6, bias=True)
    )
    
    
    model = LSTMPlus(dls.vars, dls.c, dls.len, last_step=False)
    learn = Learner(dls, model,  metrics=accuracy)
    learn.fit_one_cycle(1, 3e-3)__

0.00% [0/1 00:00<?] 

epoch | train_loss | valid_loss | accuracy | time  
---|---|---|---|---  
  
0.00% [0/11 00:00<?] 
    
    
    custom_head = partial(create_pool_head, concat_pool=True)
    model = LSTMPlus(dls.vars, dls.c, dls.len, last_step=False, custom_head=custom_head)
    learn = Learner(dls, model,  metrics=accuracy)
    learn.fit_one_cycle(1, 3e-3)__
    
    
    custom_head = partial(create_pool_plus_head, concat_pool=True)
    model = LSTMPlus(dls.vars, dls.c, dls.len, last_step=False, custom_head=custom_head)
    learn = Learner(dls, model,  metrics=accuracy)
    learn.fit_one_cycle(1, 3e-3)__
    
    
    m = RNNPlus(c_in, c_out, seq_len, hidden_size=100,n_layers=2,bidirectional=True,rnn_dropout=.5,fc_dropout=.5)
    print(m)
    print(count_parameters(m))
    m(xb).shape __
    
    
    RNNPlus(
      (backbone): _RNN_Backbone(
        (to_cat_embed): Identity()
        (feature_extractor): Identity()
        (rnn): Sequential(
          (0): RNN(3, 100, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
          (1): LSTMOutput()
        )
        (transpose): Transpose(dims=-1, -2).contiguous()
      )
      (head): Sequential(
        (0): LastStep()
        (1): Dropout(p=0.5, inplace=False)
        (2): Linear(in_features=200, out_features=2, bias=True)
      )
    )
    81802
    
    
    torch.Size([16, 2])
    
    
    m = LSTMPlus(c_in, c_out, seq_len, hidden_size=100,n_layers=2,bidirectional=True,rnn_dropout=.5,fc_dropout=.5)
    print(m)
    print(count_parameters(m))
    m(xb).shape __
    
    
    LSTMPlus(
      (backbone): _RNN_Backbone(
        (to_cat_embed): Identity()
        (feature_extractor): Identity()
        (rnn): Sequential(
          (0): LSTM(3, 100, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
          (1): LSTMOutput()
        )
        (transpose): Transpose(dims=-1, -2).contiguous()
      )
      (head): Sequential(
        (0): LastStep()
        (1): Dropout(p=0.5, inplace=False)
        (2): Linear(in_features=200, out_features=2, bias=True)
      )
    )
    326002
    
    
    torch.Size([16, 2])
    
    
    m = GRUPlus(c_in, c_out, seq_len, hidden_size=100,n_layers=2,bidirectional=True,rnn_dropout=.5,fc_dropout=.5)
    print(m)
    print(count_parameters(m))
    m(xb).shape __
    
    
    GRUPlus(
      (backbone): _RNN_Backbone(
        (to_cat_embed): Identity()
        (feature_extractor): Identity()
        (rnn): Sequential(
          (0): GRU(3, 100, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
          (1): LSTMOutput()
        )
        (transpose): Transpose(dims=-1, -2).contiguous()
      )
      (head): Sequential(
        (0): LastStep()
        (1): Dropout(p=0.5, inplace=False)
        (2): Linear(in_features=200, out_features=2, bias=True)
      )
    )
    244602
    
    
    torch.Size([16, 2])

## Converting a model to TorchScript
    
    
    model = GRUPlus(c_in, c_out, hidden_size=100, n_layers=2, bidirectional=True, rnn_dropout=.5, fc_dropout=.5)
    model.eval()
    inp = torch.rand(1, c_in, 50)
    output = model(inp)
    print(output)__
    
    
    tensor([[-0.0677, -0.0857]], grad_fn=<AddmmBackward0>)

### Tracing
    
    
    # save to gpu, cpu or both
    traced_cpu = torch.jit.trace(model.cpu(), inp)
    print(traced_cpu)
    torch.jit.save(traced_cpu, "cpu.pt")
    
    # load cpu or gpu model
    traced_cpu = torch.jit.load("cpu.pt")
    test_eq(traced_cpu(inp), output)
    
    !rm "cpu.pt"__
    
    
    GRUPlus(
      original_name=GRUPlus
      (backbone): _RNN_Backbone(
        original_name=_RNN_Backbone
        (to_cat_embed): Identity(original_name=Identity)
        (feature_extractor): Identity(original_name=Identity)
        (rnn): Sequential(
          original_name=Sequential
          (0): GRU(original_name=GRU)
          (1): LSTMOutput(original_name=LSTMOutput)
        )
        (transpose): Transpose(original_name=Transpose)
      )
      (head): Sequential(
        original_name=Sequential
        (0): LastStep(original_name=LastStep)
        (1): Dropout(original_name=Dropout)
        (2): Linear(original_name=Linear)
      )
    )

## Converting a model to ONNX
    
    
    import onnx
    
    torch.onnx.export(model.cpu(),               # model being run
                      inp,                       # model input (or a tuple for multiple inputs)
                      "cpu.onnx",                # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      verbose=False,
                      opset_version=13,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={
                          'input'  : {0 : 'batch_size'}, 
                          'output' : {0 : 'batch_size'}} # variable length axes
                     )
    
    
    onnx_model = onnx.load("cpu.onnx")           # Load the model and check it's ok
    onnx.checker.check_model(onnx_model)__
    
    
     import onnxruntime as ort
    
    ort_sess = ort.InferenceSession('cpu.onnx')
    out = ort_sess.run(None, {'input': inp.numpy()})
    
    input_name = ort_sess.get_inputs()[0].name
    output_name = ort_sess.get_outputs()[0].name
    input_dims = ort_sess.get_inputs()[0].shape
    
    test_close(out, output.detach().numpy())
    !rm "cpu.onnx"__

  * __Report an issue


