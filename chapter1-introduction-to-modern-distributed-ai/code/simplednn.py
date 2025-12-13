import torch
import torch.nn as nn

class SimpleDNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1, bias=False):
        """
        A 3-layer network:
            x -> Linear(input_dim, hidden_dim) -> Sigmoid -> Linear(hidden_dim, output_dim)
        """
        super(SimpleDNN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.activation = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, x):
        z = self.linear1(x)
        h = self.activation(z)
        y_hat = self.linear2(h)
        return y_hat

'''
好，我们把符号统一成你说的这一套：

x → z → h → ŷ
线性 → sigmoid → 线性，一共 3 层（只看一维标量版本，没有 bias）。

前向计算（forward）

$$
z = w_1 x
$$

$$
h = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

$$
\hat{y} = w_2 h
$$

$$
L = \frac12 (y - \hat{y})^2
$$

其中

* $x$：输入
* $w_1$：第 1 层线性权重
* $w_2$：第 3 层线性权重（输出层）
* $h$：隐藏层激活
* $\sigma(\cdot)$：sigmoid，导数
  $$\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr) = h(1-h).$$

对 $w_2$ 的梯度

$$
\frac{\partial L}{\partial w_2}
===============================

\frac{\partial L}{\partial \hat{y}},
\frac{\partial \hat{y}}{\partial w_2}
=====================================

(\hat{y}-y),h.
$$

对 $w_1$ 的梯度（完整链式）

$$
\frac{\partial L}{\partial w_1}
===============================

\frac{\partial L}{\partial \hat{y}},
\frac{\partial \hat{y}}{\partial h},
\frac{\partial h}{\partial z},
\frac{\partial z}{\partial w_1}.
$$

逐项：

* $\displaystyle \frac{\partial L}{\partial \hat{y}} = \hat{y}-y$
* $\displaystyle \frac{\partial \hat{y}}{\partial h} = w_2$
* $\displaystyle \frac{\partial h}{\partial z} = \sigma'(z) = h(1-h)$
* $\displaystyle \frac{\partial z}{\partial w_1} = x$

所以

$$
\frac{\partial L}{\partial w_1}
===============================

(\hat{y}-y),w_2,h(1-h),x.
$$

总结一下（你这个 3-layer DNN 的标量版 backprop 结果）：

$$
\boxed{\frac{\partial L}{\partial w_2} = (\hat{y}-y),h}
$$

$$
\boxed{\frac{\partial L}{\partial w_1} = (\hat{y}-y),w_2,h(1-h),x}
$$

'''