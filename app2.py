import xgboost as xgb
import sklearn.datasets
import sklearn.metrics
import sklearn.feature_selection
import sklearn.feature_extraction
import sklearn.model_selection
import tqdm

import matplotlib.pyplot as plt
import math 
import numpy as np
import random

xN = 3
count = 10000
X = [None] * count
y = [None] * count

def correct_func(x):
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
    y = model.predict(xgb.DMatrix(x))

    for i in range(count):
        plt_x[i] = x[i][0]
        plt_y[i] = y[i]
    plt.plot(plt_x, plt_y)

for i in range(10000):
    X[i] = np.array([2 * random.random() - 1 for _ in range(xN)], dtype=float)
    y[i] = correct_func(X[i]) + 0.1 * random.random()

sep = int(count * 0.9)
x_tr = X[0 : sep]
x_te = X[sep:]
y_tr = y[0 : sep]
y_te = y[sep:]

print("ytr", y_tr[:10])

batch_size = 10
iterations = 100
model = None
for i in range(iterations):
    for start in range(0, len(x_tr), batch_size):
        model = xgb.train({
            'learning_rate': 0.007,
            # 'update':'refresh',
            # 'process_type': 'update',
            # 'refresh_leaf': True,
            #'reg_lambda': 3,  # L2
            'reg_alpha': 3,  # L1
            'silent': False,
            'max_depth': 3,
        }, dtrain=xgb.DMatrix(x_tr[start:start+batch_size], y_tr[start:start+batch_size]), xgb_model=model)

        y_pr = model.predict(xgb.DMatrix(x_te))
        #print('    MSE itr@{}: {}'.format(int(start/batch_size), sklearn.metrics.mean_squared_error(y_te, y_pr)))
    print('MSE itr@{}: {}'.format(i, sklearn.metrics.mean_squared_error(y_te, y_pr)))
    plot_correct_func()
    plot_learned_func(model)
    plt.show()

y_pr = model.predict(xgb.DMatrix(x_te))
print('MSE at the end: {}'.format(sklearn.metrics.mean_squared_error(y_te, y_pr)))

plot_correct_func()
plot_learned_func(model)
plt.show()
