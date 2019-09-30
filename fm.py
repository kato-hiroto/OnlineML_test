import matplotlib.pyplot as plt
import math 
import numpy as np
import random, time

# 次元数など
xN = 3
yN = 1
term_num = 15       # 基底関数での説明変数一つ当たりの項数
batch_size = 10     # バッチ学習でのバッチの大きさ

# 正解の関数
def correct_func(x):
    # # 式 y = cos(dot(w, x))
    # # 重みを適当に決定
    # w = np.array([math.sin((i + 1) * 607.0 / 991.0) for i in range(xN)], dtype=float)
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
    plt.plot(plt_x, plt_y)

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
    plt.plot(plt_x, plt_y)

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
            ret_vec[3 * i + j] = math.exp(- ((x[i] - mu) ** 2) / (2 * (s ** 2)))

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
    eta = 0.95 / np.dot(phi, phi)
    r   = 0.01 * W
    new_W = W + eta * (t - y) * phi - r
    return y, new_W

# 学習プロセス
def learning(W):
    # 学習回数の決定
    count = 1
    # データ対の保存場所
    try_y    = np.array([0] * count, dtype=float)
    learn_y  = np.array([0] * count, dtype=float)
    teach_t  = np.array([0] * count, dtype=float)
    # 学習
    for i in range(count):
        # 入力と正解のデータを適当に生成
        x   = np.array([2 * random.random() - 1 for _ in range(xN)], dtype=float)
        phi = basis_func(x)
        t   = correct_func(x) + 0.1 * random.random()
        # 学習器に読ませる
        y, W = online_func_learning(phi, t, W)
        # データ対の保存
        try_y[i]   = y
        teach_t[i] = t
        learn_y[i] = online_func_forward(phi, W)
    # プロット
    # cutoff = 40
    # plt.scatter(np.array(range(cutoff, count)), ((teach_t - try_y))[cutoff:], s=8)
    # plt.show()
    # plt.scatter(np.array(range(count-100, count)), teach_t[count-100:], s=10)
    # plt.plot(np.array(range(count-100, count)), learn_y[count-100:], color="red")
    # plt.show()
    return W

# オンラインバッチ学習器 逆方向含む学習
def online_func_batch_learning(phi, t, W):
    y   = [ online_func_forward(phi[i], W)  for i in range(batch_size) ]
    eta = [ 0.1 / np.dot(phi[i], phi[i])    for i in range(batch_size) ]
    r   = 0.00 * W
    # バッチの総和
    sum_e = None
    for i in range(batch_size):
        if sum_e is None:
            sum_e = eta[i] * (t[i] - y[i]) * phi[i]
        else:
            sum_e += eta[i] * (t[i] - y[i]) * phi[i]
    # 更新
    new_W = W + 1 / batch_size * sum_e - r
    return y, new_W

# ミニバッチ学習プロセス
def batch_learning(W):
    # 学習回数の決定
    count = 100000
    # データ対の保存場所
    try_y    = np.array([0] * count, dtype=float)
    learn_y  = np.array([0] * count, dtype=float)
    teach_t  = np.array([0] * count, dtype=float)
    # 学習
    for i in range(count):
        # 入力と正解のデータを適当に生成
        x   = [ np.array([2 * random.random() - 1 for _ in range(xN)], dtype=float) for _ in range(batch_size) ]
        phi = [ basis_func(x[i])                                                    for i in range(batch_size) ]
        t   = [ correct_func(x[i]) + 0 * random.random()                            for i in range(batch_size) ]
        # 学習器に読ませる
        y, W = online_func_batch_learning(phi, t, W)
        # データ対の保存
        try_y[i]   = y[0]
        teach_t[i] = t[0]
        learn_y[i] = online_func_forward(phi[0], W)
    # プロット
    cutoff = 40
    plt.scatter(np.array(range(cutoff, count)), ((teach_t - try_y))[cutoff:], s=8)
    plt.show()
    # plt.scatter(np.array(range(count-100, count)), teach_t[count-100:], s=10)
    # plt.plot(np.array(range(count-100, count)), learn_y[count-100:], color="red")
    # plt.show()
    return W

# 実行
if __name__ == "__main__":
    # 学習する重みWを用意して学習
    learn_W = np.array([random.random() for _ in range(xN * term_num + 1)], dtype=float)
    start = time.time()
    learn_W = batch_learning(learn_W)
    end   = time.time()
    # 関数の概形表示
    print("経過時間", (end - start), "[s]")
    plot_correct_func()
    plot_learned_func(learn_W)
    plt.show()
