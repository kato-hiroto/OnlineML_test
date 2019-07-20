import matplotlib.pyplot as plt
import math 
import numpy as np
import random

# 次元数など
xN = 5
yN = 1
term_num = 7    # 基底関数での説明変数一つ当たりの項数

# 正解の関数
def correct_func(x):
    # 重みを適当に決定
    w = np.array([math.sin((i + 1) * 607.0 / 991.0) for i in range(xN)], dtype=float)
    # 式 y = cos(dot(w, x))
    y = math.cos(0.5 * math.pi * np.dot(w, x))
    return y

# 関数の描画 説明変数はx=0のみ
def plot_correct_func():
    # プロット
    count = 100
    plt_x = np.array([0] * count, dtype=float)
    plt_y = np.array([0] * count, dtype=float)
    for i in range(count):
        x = np.array([i * 0.02 - 1] * xN, dtype=float)
        y = correct_func(x)
        plt_x[i] = x[0]
        plt_y[i] = y
    plt.scatter(plt_x, plt_y)
    plt.show()

# 基底関数 xはベクトル
def basis_func(x):
    # 入力を結合する
    ret_vec  = [0] * (term_num * xN + 1)
    for i in range(xN):
        for j in range(term_num):
            # 多項式近似
            # ret_vec[3 * i + j] = x[i] ** (j + 1)
            # ガウス基底
            mu = 0.25 * i - 0.75
            s  = 0.25
            math.exp(- ((x[i] - mu) ** 2) / (2 * (s ** 2)))
    # 最後にバイアスの項
    ret_vec[-1] = 1
    return np.array(ret_vec, dtype=float).T

# オンライン学習器 順方向
def online_func_forward(phi, W):
    # 式 y = Wφ
    return np.dot(W, phi)

# オンライン学習器 逆方向含む学習
def online_func_learning(phi, t, W):
    # ∇E = sum(tφ) - sum(wφ^2) + λw = sum((t - wφ)φ)
    # 式 W' = W + η(t - Wφ)φ
    # η = 1 / dot(φ, φ) なら η(t - Wφ)φ = (t - y) * φT / dot(φ, φ)
    y   = online_func_forward(phi, W)
    eta = 0.8 / np.dot(phi, phi)
    r   = 0.3 * W
    new_W = W + eta * (t - y) * phi - r
    return y, new_W

# 学習プロセス
def learning(W):
    # プロット回数の決定
    count = 1000
    # データ対の保存場所
    try_y    = np.array([0] * count, dtype=float)
    teach_t  = np.array([0] * count, dtype=float)
    # 学習
    for i in range(count):
        # 入力と正解のデータを適当に生成
        x   = np.array([1 * random.random() - 0.5 for _ in range(xN)], dtype=float)
        phi = basis_func(x)
        t   = correct_func(x) + 0.5 * random.random()
        # 学習器に読ませる
        y, W = online_func_learning(phi, t, W)
        # データ対の保存
        try_y[i]   = y
        teach_t[i] = t
    # プロット
    cutoff = 40
    plt.scatter(np.array(range(cutoff, count)), (teach_t - try_y)[cutoff:] ** 2, s=8)
    plt.show()

# 実行
if __name__ == "__main__":
    plot_correct_func()
    # 学習する重みWを用意して学習
    learn_W = np.array([random.random() for _ in range(xN * term_num + 1)], dtype=float)
    learning(learn_W)
