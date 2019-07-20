import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import random

# バギング＆ブースト
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# loaded_data = load_boston()
# X_train, X_test, y_train, y_test = train_test_split(loaded_data["data"], loaded_data["target"], random_state=0)

xN = 1000
count = 10000
X = [None] * count
y = [None] * count

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
def plot_learned_func(model):
    # プロット
    count = 100
    plt_x = np.array([0] * count, dtype=float)
    plt_y = np.array([0] * count, dtype=float)

    x = [None] * count
    for i in range(count):
        x[i] = np.array([i * 0.02 - 1] * xN, dtype=float)
    y = model.predict(np.array(x, dtype=float))

    for i in range(count):
        plt_x[i] = x[i][0]
        plt_y[i] = y[i]
    plt.plot(plt_x, plt_y)

for i in range(10000):
    X[i] = np.array([2 * random.random() - 1 for _ in range(xN)], dtype=float)
    y[i] = correct_func(X[i]) + 0.1 * random.random()

sep = int(count * 0.9)

X_train = X[0 : sep]
X_test  = X[sep:]
y_train = y[0 : sep]
y_test  = y[sep:]

models = {
    'GradientBoost': GradientBoostingRegressor(random_state=0)
}

scores = {}
for model_name, model in models.items():
    for i in range(len(y_train)):
        if i > 0:
            p = model.get_params()
            print(p)
            # model.set_params(p)
            model.fit([X_train[i]], [y_train[i]])
        else:
            model.fit([X_train[i]], [y_train[i]])

    scores[(model_name, 'train_score')] = model.score(X_train, y_train)
    scores[(model_name, 'test_score')]  = model.score(X_test, y_test)

print(pd.Series(scores).unstack())

plot_correct_func()
plot_learned_func(model)
plt.show()
