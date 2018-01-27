# ModuleEXT-mx
ModuleEXT-mx is the extended version of mxnet.module.Module, it's based on MXNet.

It supports for L2-Norm Gradients Clipping, printing the gradients for each parameters in the network, setting optimizer states before initializing the optimizer and so on.

# Extended API

- L2-Norm Gradients Clipping
```python
set_l2norm_grad_clip(self, clip_gradients = 35, clip_gradients_global = True, verbose = False)
```

- Setting optimizer states before initializing the optimizer
```python
set_preload_optimizer_states(self, fname = None, prefix = None, epoch = None)
```
