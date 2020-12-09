# disentanglement_lib-PyTorch!
![Sample visualization](https://github.com/google-research/disentanglement_lib/blob/master/sample.gif?raw=true)

**disentanglement_lib-PyTorch** is an unofficial implementation of  google's [disentanglement_lib](https://github.com/google-research/disentanglement_lib).

Main contributions:

1. Shift Tensorflow to PyTorch! I would say, "PyTorch Yes!"
2. Decouple several modules. Now, only method parts depend on the deep learning framework. (Except the visualization part.) I hope this library supports more frameworks in the future.
3. Basically, I have tried my best to keep the original codes unchanged, though it does cause a higher time consumption.
4. Remove other parts unrelated to the unsupervised setting. This is not a feature at all, but I don't want to waste time on that. 