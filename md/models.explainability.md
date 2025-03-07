## On this page

  * get_attribution_map
  * get_acts_and_grads



  * __Report an issue



# Explainability

> Functionality to help with both global and local explainability.

* * *

source

### get_attribution_map

> 
>      get_attribution_map (model, modules, x, y=None, detach=True, cpu=False,
>                           apply_relu=True)

* * *

source

### get_acts_and_grads

> 
>      get_acts_and_grads (model, modules, x, y=None, detach=True, cpu=False)

_Returns activations and gradients for given modules in a model and a single input or a batch. Gradients require y value(s). If they are not provided, it will use the predictions._

  * __Report an issue


