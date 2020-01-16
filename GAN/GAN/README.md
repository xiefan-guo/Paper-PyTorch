# GAN

PyTorch implementation of "Generative Adversarial Networks (NIPS 2014)" [[Paper]](https://arxiv.org/abs/1406.2661).

**Authors**: _Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio_


### Loss functions

$$
\min_{G}\max_{D}V(G,D) = E_{x\sim p_{data}(x)}\left[ \log D(x) \right] + E_{x\sim p_z(z)}\left[ \log(1-D(G(z))) \right]\\\\
\min_{G}\max_{D}V(G,D) = E_{x\sim p_{data}(x)}\left[ \log D(x) \right] + E_{x\sim p_z(z)}\left[- \log D(G(z)) \right]
$$

### Usage

#### Train model

```bash
python gan.py
```