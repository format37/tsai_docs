## On this page

  * GRU
  * LSTM
  * RNN
  * Converting a model to TorchScript
    * Tracing
    * Scripting
  * Converting a model to ONNX



  * __Report an issue



  1. Models
  2. RNNs
  3. RNNs



# RNNs

> These are RNN, LSTM and GRU PyTorch implementations created by Ignacio Oguiza - oguiza@timeseriesAI.co

* * *

source

### GRU

> 
>      GRU (c_in, c_out, hidden_size=100, n_layers=1, bias=True, rnn_dropout=0,
>           bidirectional=False, fc_dropout=0.0, init_weights=True)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### LSTM

> 
>      LSTM (c_in, c_out, hidden_size=100, n_layers=1, bias=True, rnn_dropout=0,
>            bidirectional=False, fc_dropout=0.0, init_weights=True)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_

* * *

source

### RNN

> 
>      RNN (c_in, c_out, hidden_size=100, n_layers=1, bias=True, rnn_dropout=0,
>           bidirectional=False, fc_dropout=0.0, init_weights=True)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    bs = 16
    c_in = 3
    seq_len = 12
    c_out = 2
    xb = torch.rand(bs, c_in, seq_len)
    test_eq(RNN(c_in, c_out, hidden_size=100, n_layers=2, bias=True, rnn_dropout=0.2, bidirectional=True, fc_dropout=0.5)(xb).shape, [bs, c_out])
    test_eq(RNN(c_in, c_out)(xb).shape, [bs, c_out])
    test_eq(RNN(c_in, c_out, hidden_size=100, n_layers=2, bias=True, rnn_dropout=0.2, bidirectional=True, fc_dropout=0.5)(xb).shape, [bs, c_out])
    test_eq(LSTM(c_in, c_out)(xb).shape, [bs, c_out])
    test_eq(GRU(c_in, c_out)(xb).shape, [bs, c_out])__
    
    
    from tsai.basics import *__
    
    
    dsid = 'NATOPS' 
    bs = 16
    X, y, splits = get_UCR_data(dsid, return_split=False)
    tfms  = [None, [TSCategorize()]]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
    dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=bs, num_workers=0, shuffle=False)
    model = LSTM(dls.vars, dls.c)
    learn = Learner(dls, model,  metrics=accuracy)
    learn.fit_one_cycle(1, 3e-3)__

epoch | train_loss | valid_loss | accuracy | time  
---|---|---|---|---  
0 | 1.743440 | 1.633068 | 0.361111 | 00:01  
      
    
    m = RNN(c_in, c_out, hidden_size=100,n_layers=2,bidirectional=True,rnn_dropout=.5,fc_dropout=.5)
    print(m)
    print(count_parameters(m))
    m(xb).shape __
    
    
    RNN(
      (rnn): RNN(3, 100, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
      (dropout): Dropout(p=0.5, inplace=False)
      (fc): Linear(in_features=200, out_features=2, bias=True)
    )
    81802
    
    
    torch.Size([16, 2])
    
    
    m = LSTM(c_in, c_out, hidden_size=100,n_layers=2,bidirectional=True,rnn_dropout=.5,fc_dropout=.5)
    print(m)
    print(count_parameters(m))
    m(xb).shape __
    
    
    LSTM(
      (rnn): LSTM(3, 100, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
      (dropout): Dropout(p=0.5, inplace=False)
      (fc): Linear(in_features=200, out_features=2, bias=True)
    )
    326002
    
    
    torch.Size([16, 2])
    
    
    m = GRU(c_in, c_out, hidden_size=100,n_layers=2,bidirectional=True,rnn_dropout=.5,fc_dropout=.5)
    print(m)
    print(count_parameters(m))
    m(xb).shape __
    
    
    GRU(
      (rnn): GRU(3, 100, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
      (dropout): Dropout(p=0.5, inplace=False)
      (fc): Linear(in_features=200, out_features=2, bias=True)
    )
    244602
    
    
    torch.Size([16, 2])

## Converting a model to TorchScript
    
    
    model = LSTM(c_in, c_out, hidden_size=100, n_layers=2, bidirectional=True, rnn_dropout=.5, fc_dropout=.5)
    model.eval()
    inp = torch.rand(1, c_in, 50)
    output = model(inp)
    print(output)__
    
    
    tensor([[-0.0287, -0.0105]], grad_fn=<AddmmBackward0>)

### Tracing
    
    
    # save to gpu, cpu or both
    traced_cpu = torch.jit.trace(model.cpu(), inp)
    print(traced_cpu)
    torch.jit.save(traced_cpu, "cpu.pt")
    
    # load cpu or gpu model
    traced_cpu = torch.jit.load("cpu.pt")
    test_eq(traced_cpu(inp), output)
    
    !rm "cpu.pt"__
    
    
    LSTM(
      original_name=LSTM
      (rnn): LSTM(original_name=LSTM)
      (dropout): Dropout(original_name=Dropout)
      (fc): Linear(original_name=Linear)
    )

### Scripting
    
    
    # save to gpu, cpu or both
    scripted_cpu = torch.jit.script(model.cpu())
    print(scripted_cpu)
    torch.jit.save(scripted_cpu, "cpu.pt")
    
    # load cpu or gpu model
    scripted_cpu = torch.jit.load("cpu.pt")
    test_eq(scripted_cpu(inp), output)
    
    !rm "cpu.pt"__
    
    
    RecursiveScriptModule(
      original_name=LSTM
      (rnn): RecursiveScriptModule(original_name=LSTM)
      (dropout): RecursiveScriptModule(original_name=Dropout)
      (fc): RecursiveScriptModule(original_name=Linear)
    )

## Converting a model to ONNX
    
    
    import onnx
    
    # Export the model
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
    
    # Load the model and check it's ok
    onnx_model = onnx.load("cpu.onnx")
    onnx.checker.check_model(onnx_model)
    
    # You can ignore the WARNINGS below __
    
    
     import onnxruntime as ort
    
    ort_sess = ort.InferenceSession('cpu.onnx')
    out = ort_sess.run(None, {'input': inp.numpy()})
    
    # input & output names
    input_name = ort_sess.get_inputs()[0].name
    output_name = ort_sess.get_outputs()[0].name
    
    # input dimensions
    input_dims = ort_sess.get_inputs()[0].shape
    print(input_name, output_name, input_dims)
    
    test_close(out, output.detach().numpy())
    !rm "cpu.onnx"__

  * __Report an issue


