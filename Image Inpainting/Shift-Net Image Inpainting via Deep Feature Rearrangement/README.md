# Shift-Net

PyTorch implementation of "Shift-Net: Image Inpainting via Deep Feature Rearrangement (ECCV 2018)" [[Paper]](https://arxiv.org/abs/1801.09392).

**Authors**: _Zhaoyi Yan, Xiaoming Li, Mu Li, Wangmeng Zuo, Shiguang Shan_

In this paper, we introduce a special shift-connection layer to the U-Net architecture, namely Shift-Net , for filling in missing regions of any shape with sharp structures and fine-detailed textures. To this end, the encoder feature of the known region is shifted to serve as an estimation of the missing parts.

A guidance loss is introduced on decoder feature to minimize the distance between the decoder feature after fully connected layer and the ground-truth encoder feature of the missing parts.

![](https://github.com/Zhaoyi-Yan/Shift-Net_pytorch/blob/4e611ef928fdeb52d4fd5521a6f2b2d3bf3fb4a5/architecture.png)