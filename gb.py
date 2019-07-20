import xgboost as xgb
import sklearn.datasets
import sklearn.metrics
import sklearn.feature_selection
import sklearn.feature_extraction
import sklearn.model_selection
import tqdm

df = sklearn.datasets.load_boston()
print(df.keys())
print(df['feature_names'])

X = df['data']
y = df['target']

x_tr, x_te, y_tr, y_te = sklearn.model_selection.train_test_split(df["data"], df["target"], random_state=0)

print("ytr", y_tr[:10])

# one_shot_model = xgb.train({
#     # 'update':'refresh',
#     # 'process_type': 'update',
#     # 'refresh_leaf': True,
#     'silent': False,
# }, dtrain=xgb.DMatrix(x_tr, y_tr))
# y_pr = one_shot_model.predict(xgb.DMatrix(x_te))
# sklearn.metrics.mean_squared_error(y_te, y_pr)

batch_size = 50
iterations = 25
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
        }, dtrain=xgb.DMatrix(x_tr[start:start+batch_size], y_tr[start:start+batch_size]), xgb_model=model)

        y_pr = model.predict(xgb.DMatrix(x_te))
        #print('    MSE itr@{}: {}'.format(int(start/batch_size), sklearn.metrics.mean_squared_error(y_te, y_pr)))
    print('MSE itr@{}: {}'.format(i, sklearn.metrics.mean_squared_error(y_te, y_pr)))

y_pr = model.predict(xgb.DMatrix(x_te))
print('MSE at the end: {}'.format(sklearn.metrics.mean_squared_error(y_te, y_pr)))

# import pandas as pd
# import math
# import numpy as np
# import matplotlib.pyplot as plt

# # バギング＆ブースト
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.ensemble import BaggingRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# loaded_data = load_boston()
# X_train, X_test, y_train, y_test = train_test_split(loaded_data["data"], loaded_data["target"], random_state=0)

# models = {
#     'GradientBoost': GradientBoostingRegressor(random_state=0)
# }

# scores = {}
# for model_name, model in models.items():
#     for i in range(len(y_train)):
#         if i > 0:
#             p = model.get_params()
#             print(p)
#             # model.set_params(p)
#             model.fit([X_train[i]], [y_train[i]])
#         else:
#             model.fit([X_train[i]], [y_train[i]])

#     scores[(model_name, 'train_score')] = model.score(X_train, y_train)
#     scores[(model_name, 'test_score')]  = model.score(X_test, y_test)

# print(pd.Series(scores).unstack())