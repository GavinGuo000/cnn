"""
train.py —— MNIST 训练脚本
============================
下载 MNIST 数据集，训练 mini_cnn，并输出每个 epoch 的 loss 和准确率。

用法：
    python train.py

依赖：
    pip install numpy
    （数据下载仅用标准库 urllib，无需额外安装）
"""

import os
import gzip
import struct
import urllib.request
import time
import numpy as np

from mini_cnn import MiniCNN

# ============================================================
# MNIST 数据加载
# ============================================================

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def download_mnist():
    """下载 MNIST 数据集到 data/ 目录。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    for key, filename in MNIST_FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            url = MNIST_URL + filename
            print(f"  下载 {filename} ...")
            urllib.request.urlretrieve(url, filepath)
    print("  数据下载完成。")


def load_images(filename):
    """从 gzip 文件加载 MNIST 图像。"""
    filepath = os.path.join(DATA_DIR, filename)
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
    return images


def load_labels(filename):
    """从 gzip 文件加载 MNIST 标签。"""
    filepath = os.path.join(DATA_DIR, filename)
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def prepare_data():
    """
    加载并预处理 MNIST 数据。

    Returns
    -------
    X_train : (N, 1, 28, 28) float32, 归一化到 [0, 1]
    y_train : (N,) int
    X_test  : (M, 1, 28, 28) float32
    y_test  : (M,) int
    """
    print("[1/4] 准备 MNIST 数据...")
    download_mnist()

    X_train = load_images(MNIST_FILES["train_images"])
    y_train = load_labels(MNIST_FILES["train_labels"])
    X_test = load_images(MNIST_FILES["test_images"])
    y_test = load_labels(MNIST_FILES["test_labels"])

    # 归一化到 [0, 1]，并增加通道维度 -> (N, 1, 28, 28)
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    print(f"  训练集: {X_train.shape}, 测试集: {X_test.shape}")
    return X_train, y_train, X_test, y_test


# ============================================================
# 训练循环
# ============================================================

def train(epochs=5, batch_size=64, lr=0.01):
    """
    训练 MiniCNN。

    Parameters
    ----------
    epochs : int
        训练轮数
    batch_size : int
        批大小
    lr : float
        学习率
    """
    X_train, y_train, X_test, y_test = prepare_data()

    print(f"\n[2/4] 初始化模型 (lr={lr}, batch_size={batch_size})")
    model = MiniCNN()
    model.optimizer.lr = lr

    N = X_train.shape[0]
    print(f"\n[3/4] 开始训练 ({epochs} epochs)...\n")
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Train Acc':>10} | {'Test Acc':>9} | {'Time':>6}")
    print("-" * 62)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # 打乱训练集
        indices = np.random.permutation(N)
        X_train = X_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0.0
        num_batches = 0

        # mini-batch 训练
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            # 前向 → 损失 → 反向 → 更新
            logits = model.forward(X_batch)
            loss = model.compute_loss(logits, y_batch)
            model.backward()
            model.update()

            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # 计算准确率（取子集加速）
        train_acc = model.accuracy(X_train[:2000], y_train[:2000])
        test_acc = model.accuracy(X_test[:2000], y_test[:2000])

        elapsed = time.time() - t0
        print(f"{epoch:>6} | {avg_loss:>8.4f} | {train_acc:>10.2%} | {test_acc:>9.2%} | {elapsed:>5.1f}s")

    # 最终全量测试
    print(f"\n[4/4] 最终测试集准确率:")
    final_acc = model.accuracy(X_test, y_test)
    print(f"  Test Accuracy = {final_acc:.2%}  ({int(final_acc * len(y_test))}/{len(y_test)})")

    # 保存模型参数
    save_model(model)

    return model


def save_model(model):
    """将模型参数保存到 .npz 文件。"""
    params = {}
    for i, layer in enumerate(model.layers):
        for name, param, _ in layer.params():
            params[f"layer{i}_{name}"] = param
    save_path = os.path.join(DATA_DIR, "mini_cnn_params.npz")
    np.savez(save_path, **params)
    print(f"\n  模型参数已保存到: {save_path}")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print("=" * 62)
    print("  Mini-CNN  ——  纯 NumPy 手写卷积神经网络")
    print("  数据集: MNIST 手写数字 (28×28, 10 类)")
    print("=" * 62)
    print()

    model = train(
        epochs=5,      # 训练 5 轮（学习用，可自行增加）
        batch_size=64,  # 批大小
        lr=0.01,        # 学习率
    )

    print("\n训练完成！")
