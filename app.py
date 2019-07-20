import matplotlib.pyplot as plt
import math 
import numpy as np
import random

# 次元数など
xN = 1
yN = 1

# 正解の関数
def correct_func(x):
    # 重みを適当に決定
    w = np.array([math.sin((i + 1) * 607.0 / 991.0) for i in range(xN)], dtype=float)
    # 式 y = cos(dot(w, x))
    y = math.cos(np.dot(w, x))
    return y

# 基底関数
# def bf(x):
#     ret_bf = []
#     for i in range(3):

# オンライン学習器 順方向
def passive_aggressive_forward(x, W):
    # 式 y = Wx
    return np.dot(W, x)

# オンライン学習器 逆方向含む学習
def passive_aggressive_learning(x, t, W):
    # 式 W' = W + (t - y) * x^t / dot(x, x)
    y = passive_aggressive_forward(x, W)
    new_W = W + (t - y) * x / np.dot(x, x)
    return y, new_W

# 学習プロセス
def learning(W):
    # プロット回数の決定
    count = 100
    # データ対の保存場所
    count_x  = np.array(range(count), dtype=float)
    try_y    = np.array([0] * count, dtype=float)
    teach_t  = np.array([0] * count, dtype=float)
    # 学習
    for i in range(100):
        # 入力と正解のデータを適当に生成
        x = np.array([2 * random.random() for _ in range(xN)], dtype=float)
        t = correct_func(x) + 0.5 * random.random()
        # 学習器に読ませる
        y, W = passive_aggressive_learning(x, t, W)
        # データ対の保存
        try_y[i]   = y
        teach_t[i] = t
    # プロット
    # plt.scatter(count_x, try_y)
    plt.scatter(count_x, teach_t - try_y)
    plt.show()

# テストプロセス
def test_process(W):
    # プロット回数の決定
    count = 100
    # データ対の保存場所
    learn_x  = np.array([0] * count, dtype=float)
    try_y    = np.array([0] * count, dtype=float)
    teach_t  = np.array([0] * count, dtype=float)
    # テスト
    for i in range(count):
        # 入力と正解のデータを適当に生成
        x = np.array([2 * random.random() for _ in range(xN)], dtype=float)
        t = correct_func(x) + 0.5 * random.random()
        # 学習器に読ませる
        y = passive_aggressive_forward(x, W)
        # データ対の保存
        learn_x[i] = x[0]
        try_y[i]   = y
        teach_t[i] = t
    # プロット
    # plt.scatter(learn_x, try_y)
    # plt.scatter(learn_x, teach_t)
    # plt.show()

# 実行
if __name__ == "__main__":
    # 学習する重みWを用意して学習
    learn_W = np.array([random.random() for _ in range(xN)], dtype=float)
    learning(learn_W)
    test_process(learn_W)
