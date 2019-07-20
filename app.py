import matplotlib.pyplot as plt
import math 
import numpy as np
import random

# 次元数など
xN = 3
yN = 1
term_num = 3    # 基底関数での説明変数一つ当たりの項数

# 正解の関数
def correct_func(x):
    # 重みを適当に決定
    # w = np.array([math.sin((i + 1) * 607.0 / 991.0) for i in range(xN)], dtype=float)
    # 式 y = cos(dot(w, x))
    # y = 10 * math.cos(0.5 * math.pi * np.dot(w, x))
    # 式 y = sin(x1) + sin(x2 + pi * i / xN) + ...
    y = 0
    for i in range(xN):
        y += math.sin((2 + 0.4 * i) *  math.pi * x[i] + math.pi * i / xN)
    return y

# 正解関数の描画 説明変数はx=0のみ
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
    plt.scatter(plt_x, plt_y, s=8)
    plt.show()

# 学習した関数の描画 説明変数はx=0のみ
def plot_learned_func(W):
    # プロット
    count = 100
    plt_x = np.array([0] * count, dtype=float)
    plt_y = np.array([0] * count, dtype=float)
    for i in range(count):
        x   = np.array([i * 0.02 - 1] * xN, dtype=float)
        phi = basis_func(x)
        y   = np.dot(W, phi)
        plt_x[i] = x[0]
        plt_y[i] = y
    plt.scatter(plt_x, plt_y, s=8)
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
            param = 2 / (term_num + 1)
            mu = param * (j + 0.5 - term_num / 2)
            s  = param
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
    r   = 0.2 * W
    # r_b = np.zeros((1,xN * term_num + 1), dtype=float)
    # r_b[-1] = W[-1] * 0.6
    new_W = W + eta * (t - y) * phi - r
    return y, new_W

# 学習プロセス
def learning(W):
    # プロット回数の決定
    count = 10000
    # データ対の保存場所
    try_y    = np.array([0] * count, dtype=float)
    learn_y  = np.array([0] * count, dtype=float)
    teach_t  = np.array([0] * count, dtype=float)
    # 学習
    for i in range(count):
        # 入力と正解のデータを適当に生成
        x   = np.array([2 * random.random() - 1 for _ in range(xN)], dtype=float)
        phi = basis_func(x)
        t   = correct_func(x) + 0 * random.random()
        # 学習器に読ませる
        y, W = online_func_learning(phi, t, W)
        # データ対の保存
        try_y[i]   = y
        teach_t[i] = t
        learn_y[i] = online_func_forward(phi, W)
    # プロット
    cutoff = 40
    plt.scatter(np.array(range(cutoff, count)), ((teach_t - try_y))[cutoff:], s=8)
    plt.show()
    plt.scatter(np.array(range(count-100, count)), teach_t[count-100:], s=10)
    plt.plot(np.array(range(count-100, count)), learn_y[count-100:], color="red")
    plt.show()
    return W

# 実行
if __name__ == "__main__":
    # 関数の概形表示
    # plot_correct_func()
    # 学習する重みWを用意して学習
    learn_W = np.array([random.random() for _ in range(xN * term_num + 1)], dtype=float)
    learn_W = learning(learn_W)
    plot_learned_func(learn_W)
