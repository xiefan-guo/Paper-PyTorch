# PConv

PyTorch implementation of "Image Inpainting for Irregular Holes Using Partial Convolutions (ICLR 2018)" [[Paper]](https://arxiv.org/abs/1804.07723)

**Authors**: _Guilin Liu, Fitsum A. Reda, Kevin J. Shih, Ting-Chun Wang, Andrew Tao, Bryan Catanzaro_

Existing deep learning based image inpainting methods use a standard convolutional network over corrupted image, using convolutional filter responses conditioned on both valid pixels as well as the substitute values in the masked holes. This often leads to artifacts such as color discrepancy and blurriness. Post-processing is usually used to reduce such artifacts, but are expensive and may fail.

Liu et al. propose the use of partial convolutions, where the convolution is masked and renormalized to be conditioned on only valid pixels. Liu et al. further include a mechanism to automatically generate an updated mask for the next layer as part of the forward pass.

### Partial Convolutional Layer

#### Partial Convolution Operation

Let $\mathbf{W}$ be the convolution filter weights for the convolution filter and $b$ its the corresponding bias. $\mathbf{X}$ are the feature value (pixels values) for the current convolution (sliding) window and $\mathbf{M}$ is the corresponding binary mask.
$$
x' = \begin{cases} \mathbf{W}^T(\mathbf{X} \odot \mathbf{M})\frac{\text{sum}(\mathbf{1})}{\text{sum}(\mathbf{M})} + b, & \text{if sum}(\mathbf{M}) > 0 \\\\
0, & \text{otherwise}
\end{cases}
$$

where $\odot$ denotes element-wise multiplication, and $\mathbf{1}$ has same shape as $\mathbf{M}$ but with all elements being 1.

#### Mask Update Function

If the convolution was able to condition its output on at least one vaild input value, then we mark that location to be valid.
$$
m' = \begin{cases} 1, &\text{if sum}(\mathbf{M})>0 \\\\
0, & \text{otherwise}
\end{cases}
$$

### Network Architecture

| Module Name | Filter Size | Filters/Channels | Stride/Up Factor | BatchNorm | Nonlinearity |
| :---: | :---: | :---: | :---: | :---: | :---: |
| PConv1 | $7\times 7$ | $64$ | $2$ | - | ReLU |
| PConv2 | $5\times 5$ | $128$ | $2$ | Y | ReLU |
| PConv3 | $5\times 5$ | $256$ | $2$ | Y | ReLU |
| PConv4 | $3\times 3$ | $512$ | $2$ | Y | ReLU |
| PConv5 | $3\times 3$ | $512$ | $2$ | Y | ReLU |
| PConv6 | $3\times 3$ | $512$ | $2$ | Y | ReLU |
| PConv7 | $3\times 3$ | $512$ | $2$ | Y | ReLU |
| PConv8 | $3\times 3$ | $512$ | $2$ | Y | ReLU |
| NearestUpSample1 <br> Concat1(w/ PConv7) <br> PConv9| <br> <br> $3\times 3$  | $512$<br>$512 + 512$<br>$512$ | $2$<br><br>$1$ |-<br>-<br>Y | -<br>-<br>LeakyReLU(0.2) |
| NearestUpSample2 <br> Concat2(w/ PConv6) <br> PConv10| <br> <br> $3\times 3$  | $512$<br>$512 + 512$<br>$512$ | $2$<br><br>$1$ |-<br>-<br>Y | -<br>-<br>LeakyReLU(0.2) |
| NearestUpSample3 <br> Concat3(w/ PConv5) <br> PConv11| <br> <br> $3\times 3$  | $512$<br>$512 + 512$<br>$512$ | $2$<br><br>$1$ |-<br>-<br>Y | -<br>-<br>LeakyReLU(0.2) |
| NearestUpSample4 <br> Concat4(w/ PConv4) <br> PConv12| <br> <br> $3\times 3$  | $512$<br>$512 + 512$<br>$512$ | $2$<br><br>$1$ |-<br>-<br>Y | -<br>-<br>LeakyReLU(0.2) |
| NearestUpSample5 <br> Concat5(w/ PConv3) <br> PConv13| <br> <br> $3\times 3$  | $512$<br>$512 + 256$<br>$256$ | $2$<br><br>$1$ |-<br>-<br>Y | -<br>-<br>LeakyReLU(0.2) |
| NearestUpSample6 <br> Concat6(w/ PConv2) <br> PConv14| <br> <br> $3\times 3$  | $256$<br>$256 + 128$<br>$128$ | $2$<br><br>$1$ |-<br>-<br>Y | -<br>-<br>LeakyReLU(0.2) |
| NearestUpSample7 <br> Concat7(w/ PConv1) <br> PConv15| <br> <br> $3\times 3$  | $128$<br>$128 + 64$<br>$64$ | $2$<br><br>$1$ |-<br>-<br>Y | -<br>-<br>LeakyReLU(0.2) |
| NearestUpSample8 <br> Concat8(w/ Input) <br> PConv16| <br> <br> $3\times 3$  | $64$<br>$64 + 3$<br>$3$ | $2$<br><br>$1$ |-<br>-<br>- | -<br>-<br>- |

### Loss Functions

Given input image with hole $\mathbf{I}_{in}$, initial binary mask $\mathbf{M}$ ($0$ for holes), the network prediction $\mathbf{I}_{out}$, and the ground truth image $\mathbf{I}_{gt}$, we first define our per-pixel losses:
$$
\mathcal{L}_{hole}=\frac{1}{N_{\mathbf{I}_{gt}}}\Vert (1-\mathbf{M})\odot (\mathbf{I}_{out} - \mathbf{I}_{gt}) \Vert_1 \\\\
\mathcal{L}_{valid}=\frac{1}{N_{\mathbf{I}_{gt}}}\Vert \mathbf{M}\odot (\mathbf{I}_{out} - \mathbf{I}_{gt}) \Vert_1
$$







