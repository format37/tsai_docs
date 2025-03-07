## On this page

  * Imports
  * Positional encoders
    * PositionalEncoding
    * Coord2dPosEncoding
    * Coord1dPosEncoding



  * __Report an issue



  1. Models
  2. Transformers
  3. Positional encoders



# Positional encoders

> This includes some variations of positional encoders used with Transformers.

## Imports

## Positional encoders

* * *

source

### PositionalEncoding

> 
>      PositionalEncoding (q_len, d_model, normalize=True)
    
    
    pe = PositionalEncoding(1000, 512).detach().cpu().numpy()
    plt.pcolormesh(pe, cmap='viridis')
    plt.title('PositionalEncoding')
    plt.colorbar()
    plt.show()
    pe.mean(), pe.std(), pe.min(), pe.max(), pe.shape __

* * *

source

### Coord2dPosEncoding

> 
>      Coord2dPosEncoding (q_len, d_model, exponential=False, normalize=True,
>                          eps=0.001, verbose=False)
    
    
    cpe = Coord2dPosEncoding(1000, 512, exponential=True, normalize=True).cpu().numpy()
    plt.pcolormesh(cpe, cmap='viridis')
    plt.title('Coord2dPosEncoding')
    plt.colorbar()
    plt.show()
    plt.plot(cpe.mean(0))
    plt.show()
    plt.plot(cpe.mean(1))
    plt.show()
    cpe.mean(), cpe.std(), cpe.min(), cpe.max()__

* * *

source

### Coord1dPosEncoding

> 
>      Coord1dPosEncoding (q_len, exponential=False, normalize=True)
    
    
    cpe = Coord1dPosEncoding(1000, exponential=True, normalize=True).detach().cpu().numpy()
    plt.pcolormesh(cpe, cmap='viridis')
    plt.title('Coord1dPosEncoding')
    plt.colorbar()
    plt.show()
    plt.plot(cpe.mean(1))
    plt.show()
    cpe.mean(), cpe.std(), cpe.min(), cpe.max(), cpe.shape __
    
    
    cpe = Coord1dPosEncoding(1000, exponential=True, normalize=True).detach().cpu().numpy()
    plt.pcolormesh(cpe, cmap='viridis')
    plt.title('Coord1dPosEncoding')
    plt.colorbar()
    plt.show()
    plt.plot(cpe.mean(1))
    plt.show()
    cpe.mean(), cpe.std(), cpe.min(), cpe.max()__

  * __Report an issue


