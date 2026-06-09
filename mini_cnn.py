"""
mini_cnn.py — 纯 NumPy 手写极简卷积神经网络
=============================================
用于学习 CNN 的底层原理，不依赖任何深度学习框架。

核心组件：
  1. im2col / col2im  —— 将卷积运算转化为矩阵乘法（高效且易理解）
  2. Conv2D           —— 二维卷积层（支持 padding、stride）
  3. ReLU             —— 激活函数
  4. MaxPool2D        —— 最大池化层
  5. Flatten          —— 展平层
  6. Linear           —— 全连接层
  7. SoftmaxCrossEntropy —— Softmax + 交叉熵损失（合并计算，数值更稳定）
  8. SGD              —— 随机梯度下降优化器
  9. MiniCNN          —— 模型组装类

数据集：MNIST 手写数字 (28×28, 10 类)
"""

import numpy as np


# ============================================================
# 1. im2col / col2im —— 卷积的矩阵化加速
# ============================================================

def im2col(images, kernel_h, kernel_w, stride=1, padding=0):
    """
    将输入图像展开为列矩阵，使卷积运算变成矩阵乘法。

    Parameters
    ----------
    images : ndarray, shape (N, C, H, W)
        输入图像批次
    kernel_h, kernel_w : int
        卷积核的高和宽
    stride : int
        步幅
    padding : int
        零填充大小

    Returns
    -------
    cols : ndarray, shape (N * out_h * out_w, C * kernel_h * kernel_w)
        展开后的列矩阵
    """
    N, C, H, W = images.shape

    # 填充
    if padding > 0:
        images = np.pad(images,
                        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                        mode='constant')

    _, _, H_pad, W_pad = images.shape

    # 输出尺寸
    out_h = (H_pad - kernel_h) // stride + 1
    out_w = (W_pad - kernel_w) // stride + 1

    # 用 stride tricks 高效提取所有滑窗，再 reshape
    cols = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w))
    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            cols[:, :, y, x, :, :] = images[:, :, y:y_max:stride, x:x_max:stride]

    # reshape: (N, C, kh, kw, out_h, out_w) -> (N*out_h*out_w, C*kh*kw)
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return cols, out_h, out_w


def col2im(cols, input_shape, kernel_h, kernel_w, stride=1, padding=0, out_h=0, out_w=0):
    """
    im2col 的逆操作：将列矩阵的梯度还原回图像形状。
    用于反向传播中将梯度传回输入。

    Parameters
    ----------
    cols : ndarray, shape (N * out_h * out_w, C * kernel_h * kernel_w)
    input_shape : tuple (N, C, H, W)
    kernel_h, kernel_w : int
    stride : int
    padding : int
    out_h, out_w : int
        输出特征图的高和宽

    Returns
    -------
    images : ndarray, shape (N, C, H, W)
    """
    N, C, H, W = input_shape

    H_pad = H + 2 * padding
    W_pad = W + 2 * padding

    # 先还原为 (N, out_h, out_w, C, kh, kw)
    cols = cols.reshape(N, out_h, out_w, C, kernel_h, kernel_w)
    cols = cols.transpose(0, 3, 4, 5, 1, 2)  # (N, C, kh, kw, out_h, out_w)

    images = np.zeros((N, C, H_pad, W_pad))
    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            images[:, :, y:y_max:stride, x:x_max:stride] += cols[:, :, y, x, :, :]

    # 去掉 padding
    if padding > 0:
        images = images[:, :, padding:-padding, padding:-padding]

    return images


# ============================================================
# 2. Conv2D —— 二维卷积层
# ============================================================

class Conv2D:
    """
    二维卷积层。

    Parameters
    ----------
    in_channels : int
        输入通道数
    out_channels : int
        输出通道数（卷积核个数）
    kernel_size : int
        卷积核边长（正方形核）
    stride : int
        步幅，默认 1
    padding : int
        零填充，默认 0
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He 初始化（适合 ReLU）
        fan_in = in_channels * kernel_size * kernel_size
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) \
                 * np.sqrt(2.0 / fan_in)
        self.b = np.zeros((out_channels, 1))

        # 梯度占位
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        前向传播：im2col + 矩阵乘法

        x : (N, C_in, H, W)
        return : (N, C_out, out_h, out_w)
        """
        self.x = x
        N, C_in, H, W = x.shape

        # im2col 展开
        self.cols, self.out_h, self.out_w = im2col(
            x, self.kernel_size, self.kernel_size, self.stride, self.padding
        )
        # cols: (N*out_h*out_w, C_in*kh*kw)

        # 将卷积核 reshape 为矩阵
        # W: (C_out, C_in, kh, kw) -> (C_out, C_in*kh*kw)
        W_col = self.W.reshape(self.out_channels, -1)

        # 矩阵乘法：(N*out_h*out_w, C_in*kh*kw) @ (C_in*kh*kw, C_out) -> (N*out_h*out_w, C_out)
        out = self.cols @ W_col.T + self.b.T  # b.T: (1, C_out)

        # reshape 回特征图形状
        out = out.reshape(N, self.out_h, self.out_w, self.out_channels)
        out = out.transpose(0, 3, 1, 2)  # (N, C_out, out_h, out_w)

        return out

    def backward(self, dout):
        """
        反向传播

        dout : (N, C_out, out_h, out_w)
        return : (N, C_in, H, W)  传回上一层的梯度
        """
        N, C_out, out_h, out_w = dout.shape

        # reshape dout -> (N*out_h*out_w, C_out)
        dout_col = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        # 权重梯度：dW = dout^T @ cols -> (C_out, C_in*kh*kw)
        self.dW = (dout_col.T @ self.cols).reshape(self.W.shape)

        # 偏置梯度：db = sum(dout, axis=0) -> (C_out, 1)
        self.db = dout_col.sum(axis=0, keepdims=True).T  # (C_out, 1)

        # 输入梯度：dcols = dout @ W -> (N*out_h*out_w, C_in*kh*kw)
        W_col = self.W.reshape(self.out_channels, -1)
        dcols = dout_col @ W_col

        # col2im 还原梯度
        dx = col2im(dcols, self.x.shape, self.kernel_size, self.kernel_size,
                    self.stride, self.padding, self.out_h, self.out_w)
        return dx

    def params(self):
        return [('W', self.W, self.dW), ('b', self.b, self.db)]


# ============================================================
# 3. ReLU —— 激活函数
# ============================================================

class ReLU:
    """ReLU 激活：max(0, x)"""

    def forward(self, x):
        self.mask = (x > 0)  # 记录哪些位置 > 0，反向传播时用
        return x * self.mask

    def backward(self, dout):
        # 梯度只通过 > 0 的位置
        return dout * self.mask

    def params(self):
        return []


# ============================================================
# 4. MaxPool2D —— 最大池化层
# ============================================================

class MaxPool2D:
    """
    最大池化层。

    Parameters
    ----------
    pool_size : int
        池化窗口大小
    stride : int
        步幅，默认等于 pool_size（不重叠池化）
    """

    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size

    def forward(self, x):
        """
        前向传播：取每个窗口内的最大值。

        x : (N, C, H, W)
        return : (N, C, out_h, out_w)
        """
        self.x = x
        N, C, H, W = x.shape
        p = self.pool_size
        s = self.stride

        out_h = (H - p) // s + 1
        out_w = (W - p) // s + 1

        # 使用 im2col 展开为 (N*C*out_h*out_w, p*p)
        # 但这里手动展开更直观
        out = np.zeros((N, C, out_h, out_w))
        self.arg_max = np.zeros((N, C, out_h, out_w), dtype=int)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * s
                w_start = j * s
                window = x[:, :, h_start:h_start + p, w_start:w_start + p]
                # window: (N, C, p, p)
                window_flat = window.reshape(N, C, -1)  # (N, C, p*p)
                out[:, :, i, j] = window_flat.max(axis=2)
                self.arg_max[:, :, i, j] = window_flat.argmax(axis=2)

        self.out_h = out_h
        self.out_w = out_w
        return out

    def backward(self, dout):
        """
        反向传播：梯度只传回前向时取最大值的位置。

        dout : (N, C, out_h, out_w)
        return : (N, C, H, W)
        """
        N, C, H, W = self.x.shape
        p = self.pool_size
        s = self.stride

        dx = np.zeros_like(self.x)

        for i in range(self.out_h):
            for j in range(self.out_w):
                h_start = i * s
                w_start = j * s
                # 把 dout 的值放到前向取最大值的位置上
                for n in range(N):
                    for c in range(C):
                        idx = self.arg_max[n, c, i, j]
                        dy = idx // p
                        dx_idx = idx % p
                        dx[n, c, h_start + dy, w_start + dx_idx] += dout[n, c, i, j]

        return dx

    def params(self):
        return []


# ============================================================
# 5. Flatten —— 展平层
# ============================================================

class Flatten:
    """将 (N, C, H, W) 展平为 (N, C*H*W)"""

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)

    def params(self):
        return []


# ============================================================
# 6. Linear —— 全连接层
# ============================================================

class Linear:
    """
    全连接层：y = x @ W^T + b

    Parameters
    ----------
    in_features : int
    out_features : int
    """

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # He 初始化
        self.W = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))

        self.dW = None
        self.db = None

    def forward(self, x):
        """
        x : (N, in_features)
        return : (N, out_features)
        """
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, dout):
        """
        dout : (N, out_features)
        return : (N, in_features)
        """
        # dW = dout^T @ x -> (out_features, in_features)
        self.dW = dout.T @ self.x
        # db = sum(dout, axis=0) -> (1, out_features)
        self.db = dout.sum(axis=0, keepdims=True)
        # dx = dout @ W -> (N, in_features)
        dx = dout @ self.W
        return dx

    def params(self):
        return [('W', self.W, self.dW), ('b', self.b, self.db)]


# ============================================================
# 7. SoftmaxCrossEntropy —— Softmax + 交叉熵损失
# ============================================================

class SoftmaxCrossEntropy:
    """
    Softmax + 交叉熵损失（合并实现，数值更稳定）。

    前向：先做 softmax 归一化，再计算交叉熵
    反向：梯度 = softmax_output - one_hot_label（非常简洁）
    """

    def forward(self, x, labels):
        """
        x : (N, num_classes)  logits
        labels : (N,)  真实标签（整数）

        return : loss (标量)
        """
        N = x.shape[0]

        # 数值稳定的 softmax
        x_shifted = x - x.max(axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        self.probs = exp_x / exp_x.sum(axis=1, keepdims=True)

        # 交叉熵损失
        correct_logprobs = -np.log(self.probs[np.arange(N), labels] + 1e-8)
        loss = correct_logprobs.mean()

        self.labels = labels
        return loss

    def backward(self):
        """
        合并梯度的简洁形式：dL/dx = probs - one_hot(y)

        return : (N, num_classes)
        """
        N = self.probs.shape[0]
        dx = self.probs.copy()
        dx[np.arange(N), self.labels] -= 1
        dx /= N  # 除以 batch size
        return dx


# ============================================================
# 8. SGD —— 随机梯度下降优化器
# ============================================================

class SGD:
    """
    随机梯度下降（可选 momentum）。

    Parameters
    ----------
    lr : float
        学习率
    momentum : float
        动量系数，默认 0（不使用动量）
    """

    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}  # 记录每个参数的速度

    def step(self, layers):
        """遍历所有层的参数，执行梯度下降更新。"""
        param_id = 0
        for layer in layers:
            for name, param, grad in layer.params():
                if grad is None:
                    continue
                # momentum 速度更新
                if param_id not in self.velocities:
                    self.velocities[param_id] = np.zeros_like(param)
                v = self.velocities[param_id]
                v[:] = self.momentum * v - self.lr * grad
                param += v
                param_id += 1


# ============================================================
# 9. MiniCNN —— 模型组装
# ============================================================

class MiniCNN:
    """
    极简卷积神经网络，结构：

    Input (1, 28, 28)
      ↓
    Conv2D(1→8, 3×3, pad=1)  → (8, 28, 28)
      ↓
    ReLU                      → (8, 28, 28)
      ↓
    MaxPool2D(2×2)           → (8, 14, 14)
      ↓
    Conv2D(8→16, 3×3, pad=1) → (16, 14, 14)
      ↓
    ReLU                      → (16, 14, 14)
      ↓
    MaxPool2D(2×2)           → (16, 7, 7)
      ↓
    Flatten                   → (784,)
      ↓
    Linear(784→128)           → (128,)
      ↓
    ReLU                      → (128,)
      ↓
    Linear(128→10)            → (10,)
      ↓
    Softmax + CrossEntropy
    """

    def __init__(self):
        self.layers = [
            Conv2D(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(pool_size=2),
            Conv2D(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(pool_size=2),
            Flatten(),
            Linear(16 * 7 * 7, 128),
            ReLU(),
            Linear(128, 10),
        ]
        self.loss_fn = SoftmaxCrossEntropy()
        self.optimizer = SGD(lr=0.01, momentum=0.9)

    def forward(self, x):
        """前向传播：逐层执行。"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def compute_loss(self, logits, labels):
        """计算损失。"""
        return self.loss_fn.forward(logits, labels)

    def backward(self):
        """反向传播：从损失函数开始，逆序遍历所有层。"""
        dout = self.loss_fn.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def update(self):
        """参数更新。"""
        self.optimizer.step(self.layers)

    def predict(self, x):
        """预测类别。"""
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

    def accuracy(self, x, labels):
        """计算准确率。"""
        preds = self.predict(x)
        return np.mean(preds == labels)
