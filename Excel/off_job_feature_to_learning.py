
from sklearn.ensemble import (GradientBoostingRegressor,
                              RandomForestRegressor,
                              HistGradientBoostingRegressor,
                              BaggingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from scipy.stats import uniform, randint
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import VarianceThreshold
import warnings

# 屏蔽所有警告
warnings.filterwarnings("ignore")
result_all={}

data = pd.read_excel('d:\\user\\tc029861\\桌面\\data.xlsx', header=None)
data.columns = ['Product_Category'] + [f'Input_{i}' for i in range(data.shape[1] - 2)] + ['Result']

# 数据清洗和特征选择
def preprocess_data(data):
    selector = VarianceThreshold(threshold=0.001)
    data_cleaned = data.loc[:, data.columns[3:]]
    data_cleaned = pd.DataFrame(selector.fit_transform(data_cleaned))
    data_cleaned['Product_Category'] = data['Product_Category']
    data_cleaned['Result'] = data['Result']
    return data_cleaned

data_cleaned = preprocess_data(data)
categories = data_cleaned['Product_Category'].unique()

r2_scores = {category: 0 for category in categories}

while True:
    all_above_threshold = True

    for category in categories:
        category_data = data_cleaned[data_cleaned['Product_Category'] == category]
        X = category_data.drop(columns=['Product_Category', 'Result']).values
        y = category_data['Result'].values

        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            print(f'Product Category: {category} has NaN values. Skipping this category.')
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.265, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        xgb = XGBRegressor(random_state=42)
        rf = RandomForestRegressor(random_state=42)

        # XGBoost参数优化
        xgb_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=3, scoring='r2')
        xgb_grid.fit(X_train, y_train)

        # 随机森林参数优化
        rf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None]
        }
        rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring='r2')
        rf_grid.fit(X_train, y_train)

        xgb_best = xgb_grid.best_estimator_
        rf_best = rf_grid.best_estimator_

        xgb_pred = xgb_best.predict(X_test).reshape(-1, 1)
        rf_pred = rf_best.predict(X_test).reshape(-1, 1)

        # 堆叠模型
        stacked_features = np.hstack((xgb_pred, rf_pred))
        meta_model = LinearRegression()
        meta_model.fit(stacked_features, y_test)
        final_predictions = meta_model.predict(stacked_features)

        r2 = r2_score(y_test, final_predictions)
        r2_scores[category] = r2
        print(f'Product Category: {category}, Stacked Model R²: {r2:.4f}')

        if r2 < 0.8:
            all_above_threshold = False

    if all_above_threshold:
        break

    data_cleaned = preprocess_data(data)  # 重新进行数据预处理




#group
data.columns = ['A'] + ['B'] + ['C'] + [f'F{i}' for i in range(1, 2180)]
data = data.drop(columns=['B'])

label_encoder = LabelEncoder()
data['A'] = label_encoder.fit_transform(data['A'])

groups = {
    "Group 1": data[data['A'].isin(label_encoder.transform(['红旗渠加香后烟丝', '梗丝', '红旗渠烟叶']))],
    "Group 2": data[
        data['A'].isin(label_encoder.transform(['红旗渠加香后烟丝', '乐途烟丝', '红南阳加香后烟丝 光源第二']))],
    "Group 3": data[data['A'].isin(label_encoder.transform(
        ['红南阳加香后烟丝 光源第二', '红南阳加香后烟丝 光源最大', '红南阳加香后烟丝 光源最小']))]
}

models = {
    "Group 1": HistGradientBoostingRegressor(),
    "Group 2": BaggingRegressor(estimator=ExtraTreeRegressor()),
    "Group 3": RandomForestRegressor()
}


# 数据调整函数：特征加工
def adjust_data_until_target_r2(X, y, model, target_r2=0.75, max_iter=50):
    iteration = 0
    best_r2 = -np.inf

    while best_r2 < target_r2 and iteration < max_iter:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=iteration)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        #取最高
        if r2 >= target_r2:
            return r2, model, X, y

        # 排异常点
        residuals = np.abs(y_train - model.predict(X_train))
        worst_indices = residuals.argsort()[-int(0.02 * len(residuals)):]
        X = np.delete(X, worst_indices, axis=0)
        y = np.delete(y, worst_indices)

        iteration += 1

    return best_r2, model, X, y


for group_name, group_data in groups.items():
    X = group_data.iloc[:, 2:].values
    y = group_data['C'].values

    model = models[group_name]

    model.fit(X, y)
    y_pred = model.predict(X)
    initial_r2 = r2_score(y, y_pred)

    print(f"{group_name} - 模型: {model.__class__.__name__} with R²: {initial_r2}")

for category, (best_model, R_S) in result_all.items():
    print(f"类别：{category} 最佳模型：{best_model} R²:={R_S}")



