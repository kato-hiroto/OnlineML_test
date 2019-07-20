import matplotlib.pyplot as plt
import math 
import numpy as np
import random

# 次元数など
xN = 10
yN = 1
term_num = 3    # 基底関数での説明変数一つ当たりの項数

# 正解の関数
def correct_func(x):
    # 重みを適当に決定
    w = np.array([math.sin((i + 1) * 607.0 / 991.0) for i in range(xN)], dtype=float)
    # 式 y = cos(dot(w, x))
    y = math.cos(np.dot(w, x))
    return y

# 関数の描画 説明変数はx=0のみ
def plot_correct_func():
    # プロット
    count = 100
    plt_x = np.array([0] * count, dtype=float)
    plt_y = np.array([0] * count, dtype=float)
    for i in range(count):
        x = np.array([i * 0.4 - 2] * 10, dtype=float)
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
            ret_vec[3 * i + j] = x[i] ** (j + 1)
    # 最後にバイアスの項
    ret_vec[-1] = 1
    return np.array(ret_vec, dtype=float).T

# 損失関数
def loss_func():
    pass

# オンライン学習器 順方向
def online_func_forward(phi, W):
    # 式 y = Wφ
    return np.dot(W, phi)

# オンライン学習器 逆方向含む学習
def online_func_learning(phi, t, W):
    # 式 W' = W + η(t - Wφ)φ
    # η = 1 / dot(φ, φ) なら η(t - Wφ)φ = (t - y) * φT / dot(φ, φ)
    y   = online_func_forward(phi, W)
    eta = 0.5 / np.dot(phi, phi)
    new_W = W + eta * (t - y) * phi
    return y, new_W

# 学習プロセス
def learning(W):
    # プロット回数の決定
    count = 200
    # データ対の保存場所
    try_y    = np.array([0] * count, dtype=float)
    teach_t  = np.array([0] * count, dtype=float)
    # 学習
    for i in range(count):
        # 入力と正解のデータを適当に生成
        x   = np.array([2 * random.random() for _ in range(xN)], dtype=float)
        phi = basis_func(x)
        t   = correct_func(x) + 0.5 * random.random()
        # 学習器に読ませる
        y, W = online_func_learning(phi, t, W)
        # データ対の保存
        try_y[i]   = y
        teach_t[i] = t
    # プロット
    plt.scatter(np.array(range(count)), teach_t - try_y, s=8)
    plt.show()

# 実行
if __name__ == "__main__":
    # 学習する重みWを用意して学習
    learn_W = np.array([random.random() for _ in range(xN * term_num + 1)], dtype=float)
    learning(learn_W)
