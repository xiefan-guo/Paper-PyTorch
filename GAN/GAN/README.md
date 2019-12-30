## GAN

### 相关工作

#### KL 散度（Kullback–Leibler Divergence）
$$KL(p \Vert q)=\int p(x)log \frac{p(x)}{q(x)} = E_{x\sim p}\log \frac{p(x)}{q(x)}$$

#### JS 散度（Jensen-Shannon Divergence）
$$JS(p\Vert q)=\frac{1}{2}KL(p\Vert \frac{p+q}{2})+\frac{1}{2}KL(q\Vert \frac{p+q}{2})$$

JS 散度满足对称性，即 $JS(p\Vert q)=JS(q\Vert p)$ ；当 $p = q = 0$ 时，$JS(p\Vert q)=0$ ；当 $p = 0, q \ne 0$ 或 $p \ne q, p = q$ 时，$JS(p\Vert q) = \log 2$ 。

### GAN

* 判别器（Discriminator）&emsp;判断一个样本是真实样本还是生成器生成样本。
* 生成器（Generator）&emsp;生成样本并让判别器无法判断其是否是生成的。

#### 目标函数

$$
\min_{G}\max_{D}V(G,D) = E_{x\sim p_{data}(x)}\left[ \log D(x) \right] + E_{x\sim p_z(z)}\left[ \log(1-D(G(z))) \right]\\
\min_{G}\max_{D}V(G,D) = E_{x\sim p_{data}(x)}\left[ \log D(x) \right] + E_{x\sim p_z(z)}\left[- \log D(G(z)) \right]
$$