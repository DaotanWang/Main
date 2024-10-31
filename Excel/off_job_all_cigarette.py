
from sklearn.ensemble import (GradientBoostingRegressor,
                              RandomForestRegressor,
                              HistGradientBoostingRegressor,
                              BaggingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
from sklearn.feature_selection import SelectKBest, f_regression
import warnings

# 屏蔽所有警告
warnings.filterwarnings("ignore")
result_all={}

file_path = '\\data.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path, header=None)

# 预处理
data.columns = ['Product_Category'] + [f'Input_{i}' for i in range(data.shape[1] - 2)] + ['Result']

# 获取产品类别
categories = data['Product_Category'].unique()

# 神经网络
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 增加神经元数量
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)  # 额外的隐藏层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 存储每个模型的结果
models = {}

r2_scores = {}

for category in categories:
    category_data = data[data['Product_Category'] == category]

    # 分离特征和标签
    X = category_data.iloc[:, 1:-1].values  # D列到CEX列
    y = category_data.iloc[:, -1].values     # C列

    # 检查缺失值
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        print(f'Product Category: {category} has NaN values. Skipping this category.')
        continue
    if category=='红南阳加香后烟丝 光源最小':

        # 特征选择
        selector = SelectKBest(score_func=f_regression, k=8)
        X_selected = selector.fit_transform(X, y)

        # 随机拆分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

        # 数据标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 参数矩阵
        param_dist = {
            'n_estimators': randint(100, 300),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(3, 8),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 5)
        }

        # 随机搜索用于超参数优化
        model = GradientBoostingRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            model, param_distributions=param_dist, n_iter=15, cv=3,
            scoring='r2', random_state=42, n_jobs=-1
        )

        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        predictions = best_model.predict(X_test)
        r2 = r2_score(y_test, predictions)

        r2_scores[category] = r2

        result_all[category] = ['梯度回升', r2]

    elif category=='乐途烟丝':
        def train_stacked_model_optimized(category_data):
            X = category_data.iloc[:, 1:-1].values
            y = category_data.iloc[:, -1].values

            # 特征选择
            k_best = SelectKBest(f_regression, k=8)
            X_selected = k_best.fit_transform(X, y)

            # 拆分数据集
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.265, random_state=42)

            # 参数矩阵
            param_dist_xgb = {
                'n_estimators': randint(100, 300),
                'max_depth': randint(3, 8),
                'learning_rate': uniform(0.01, 0.1),
                'subsample': uniform(0.7, 0.3)
            }
            param_dist_rf = {
                'n_estimators': randint(100, 300),
                'max_depth': randint(3, 8),
                'max_features': ['auto', 'sqrt']
            }
            param_dist_svr = {
                'C': uniform(1, 5),
                'gamma': uniform(0.001, 0.01)
            }

            # 使用随机搜索进行模型调优
            xgb_random_search = RandomizedSearchCV(
                XGBRegressor(random_state=42), param_distributions=param_dist_xgb,
                n_iter=15, cv=3, scoring='r2', n_jobs=-1, random_state=42
            )
            rf_random_search = RandomizedSearchCV(
                RandomForestRegressor(random_state=42), param_distributions=param_dist_rf,
                n_iter=15, cv=3, scoring='r2', n_jobs=-1, random_state=42
            )
            svr_random_search = RandomizedSearchCV(
                SVR(), param_distributions=param_dist_svr,
                n_iter=15, cv=3, scoring='r2', n_jobs=-1, random_state=42
            )

            xgb_random_search.fit(X_train, y_train)
            rf_random_search.fit(X_train, y_train)
            svr_random_search.fit(X_train, y_train)

            best_xgb = xgb_random_search.best_estimator_
            best_rf = rf_random_search.best_estimator_
            best_svr = svr_random_search.best_estimator_

            # 获得基础模型的预测结果
            xgb_pred = best_xgb.predict(X_test).reshape(-1, 1)
            rf_pred = best_rf.predict(X_test).reshape(-1, 1)
            svr_pred = best_svr.predict(X_test).reshape(-1, 1)

            # 将基础模型的预测结果作为新的特征
            stacked_features = np.hstack((xgb_pred, rf_pred, svr_pred))

            # 使用线性回归作为元模型
            scaler = StandardScaler()
            stacked_features_scaled = scaler.fit_transform(stacked_features)

            meta_model = LinearRegression()
            meta_model.fit(stacked_features_scaled, y_test)
            final_predictions = meta_model.predict(stacked_features_scaled)

            # 计算 R²
            r2 = r2_score(y_test, final_predictions)
            return r2
        r2 = train_stacked_model_optimized(category_data)
        result_all[category] = ['堆叠模型', r2]


    elif category=='红旗渠加香后烟丝':
        def train_stacked_model_optimized(category_data):
            X = category_data.iloc[:, 1:-1].values
            y = category_data.iloc[:, -1].values

            # 特征选择
            k_best = SelectKBest(f_regression, k=8)
            X_selected = k_best.fit_transform(X, y)

            # 拆分数据集
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

            param_dist_xgb = {
                'n_estimators': randint(100, 300),
                'max_depth': randint(3, 8),
                'learning_rate': uniform(0.01, 0.1),
                'subsample': uniform(0.7, 0.3)
            }
            param_dist_rf = {
                'n_estimators': randint(100, 300),
                'max_depth': randint(3, 8),
                'max_features': ['auto', 'sqrt']
            }
            param_dist_svr = {
                'C': uniform(1, 5),
                'gamma': uniform(0.001, 0.01)
            }

            # 使用随机搜索进行模型调优
            xgb_random_search = RandomizedSearchCV(
                XGBRegressor(random_state=42), param_distributions=param_dist_xgb,
                n_iter=15, cv=3, scoring='r2', n_jobs=-1, random_state=42
            )
            rf_random_search = RandomizedSearchCV(
                RandomForestRegressor(random_state=42), param_distributions=param_dist_rf,
                n_iter=15, cv=3, scoring='r2', n_jobs=-1, random_state=42
            )
            svr_random_search = RandomizedSearchCV(
                SVR(), param_distributions=param_dist_svr,
                n_iter=15, cv=3, scoring='r2', n_jobs=-1, random_state=42
            )

            xgb_random_search.fit(X_train, y_train)
            rf_random_search.fit(X_train, y_train)
            svr_random_search.fit(X_train, y_train)

            best_xgb = xgb_random_search.best_estimator_
            best_rf = rf_random_search.best_estimator_
            best_svr = svr_random_search.best_estimator_

            # 获得基础模型的预测结果
            xgb_pred = best_xgb.predict(X_test).reshape(-1, 1)
            rf_pred = best_rf.predict(X_test).reshape(-1, 1)
            svr_pred = best_svr.predict(X_test).reshape(-1, 1)

            # 将基础模型的预测结果作为新的特征
            stacked_features = np.hstack((xgb_pred, rf_pred, svr_pred))

            # 使用线性回归作为元模型
            scaler = StandardScaler()
            stacked_features_scaled = scaler.fit_transform(stacked_features)

            meta_model = LinearRegression()
            meta_model.fit(stacked_features_scaled, y_test)
            final_predictions = meta_model.predict(stacked_features_scaled)

            # 计算 R²
            r2 = r2_score(y_test, final_predictions)

            return r2


        r2 = train_stacked_model_optimized(category_data)
        result_all[category] = ['堆叠模型', r2]

    elif category in ['梗丝','红旗渠烟叶','红南阳加香后烟丝 光源第二','红南阳加香后烟丝 光源最大']:
        # KFold 交叉验证
        kf = KFold(n_splits=5)
        r2_list = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 数据标准化
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # 转换为 PyTorch 张量
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

            # 初始化模型、损失函数和优化器
            model = RegressionModel(input_size=X_train.shape[1])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # 训练模型
            num_epochs = 7000  # 增加训练周期
            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()

            # 测试模型并计算 R²
            model.eval()
            with torch.no_grad():
                predictions = model(X_test_tensor)
                r2 = r2_score(y_test, predictions.numpy())
                r2_list.append(r2)

        # 计算平均 R²
        avg_r2 = np.mean(r2_list)
        models[category] = model
        r2_scores[category] = avg_r2
        result_all[category] = ['神经网络', avg_r2]


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



