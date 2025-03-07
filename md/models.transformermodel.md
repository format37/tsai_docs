## On this page

  * TransformerModel



  * __Report an issue



  1. Models
  2. Transformers
  3. TransformerModel



# TransformerModel

> This is an unofficial PyTorch implementation created by Ignacio Oguiza - oguiza@timeseriesAI.co

* * *

source

### TransformerModel

> 
>      TransformerModel (c_in, c_out, d_model=64, n_head=1, d_ffn=128,
>                        dropout=0.1, activation='relu', n_layers=1)

_Same as`nn.Module`, but no need for subclasses to call `super().__init__`_
    
    
    bs = 16
    nvars = 3
    seq_len = 96
    c_out = 2
    xb = torch.rand(bs, nvars, seq_len)
    
    
    model = TransformerModel(nvars, c_out, d_model=64, n_head=1, d_ffn=128, dropout=0.1, activation='gelu', n_layers=3)
    test_eq(model(xb).shape, [bs, c_out])
    print(count_parameters(model))
    model __
    
    
    100930
    
    
    TransformerModel(
      (permute): Permute(dims=2, 0, 1)
      (inlinear): Linear(in_features=3, out_features=64, bias=True)
      (relu): ReLU()
      (transformer_encoder): TransformerEncoder(
        (layers): ModuleList(
          (0): TransformerEncoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
            )
            (linear1): Linear(in_features=64, out_features=128, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
            (linear2): Linear(in_features=128, out_features=64, bias=True)
            (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (dropout1): Dropout(p=0.1, inplace=False)
            (dropout2): Dropout(p=0.1, inplace=False)
          )
          (1): TransformerEncoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
            )
            (linear1): Linear(in_features=64, out_features=128, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
            (linear2): Linear(in_features=128, out_features=64, bias=True)
            (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (dropout1): Dropout(p=0.1, inplace=False)
            (dropout2): Dropout(p=0.1, inplace=False)
          )
          (2): TransformerEncoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
            )
            (linear1): Linear(in_features=64, out_features=128, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
            (linear2): Linear(in_features=128, out_features=64, bias=True)
            (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (dropout1): Dropout(p=0.1, inplace=False)
            (dropout2): Dropout(p=0.1, inplace=False)
          )
        )
        (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      )
      (transpose): Transpose(1, 0)
      (max): Max(dim=1, keepdim=False)
      (outlinear): Linear(in_features=64, out_features=2, bias=True)
    )

  * __Report an issue


