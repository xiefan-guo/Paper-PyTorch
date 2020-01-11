# PConv

PyTorch implementation of "Image Inpainting for Irregular Holes Using Partial Convolutions (ICLR 2018)" [[Paper]](https://arxiv.org/abs/1804.07723)

**Authors**: _Guilin Liu, Fitsum A. Reda, Kevin J. Shih, Ting-Chun Wang, Andrew Tao, Bryan Catanzaro_

Existing deep learning based image inpainting methods use a standard convolutional network over corrupted image, using convolutional filter responses conditioned on both valid pixels as well as the substitute values in the masked holes. This often leads to artifacts such as color discrepancy and blurriness. Post-processing is usually used to reduce such artifacts, but are expensive and may fail.

Liu et al. propose the use of partial convolutions, where the convolution is masked and renormalized to be conditioned on only valid pixels. Liu et al. further include a mechanism to automatically generate an updated mask for the next layer as part of the forward pass.

### Partial Convolutional Layer

