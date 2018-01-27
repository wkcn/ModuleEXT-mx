# ModuleEXT-mx
ModuleEXT-mx is the extended version of `mxnet.module.Module`, it's based on MXNet.

It supports for the features below:
- Setting `lr_mult` or `wd_mult` without assigning idx2name for the optimizer
- L2-Norm Gradients Clipping
- Printing the gradients statistics for each parameters in the model
- Setting optimizer states before initializing the optimizer

# Extended API

**class ModuleEXT(mxnet.module.Module)**
- L2-Norm Gradients Clipping
```python
set_l2norm_grad_clip(clip_gradients = 35, clip_gradients_global = True, verbose = False, used = True)
```

- Setting optimizer states before initializing the optimizer
```python
set_preload_optimizer_states(fname = None, prefix = None, epoch = None)
```

- Printing the gradients statistics for each parameters in the model (only print the gradients on the device 0)
```python
print_gradients(names = None)
```
If names is None, printing all gradients statistics.
Otherwise, printing the specific gradients statistics according to the list of the names.

- Getting the indexes according to the names of the parameters. 
```python
get_idxes_from_names(names)
```
