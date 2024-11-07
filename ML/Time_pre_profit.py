import pandas as pd
from prophet import Prophet
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel('your_file.xlsx')  # 请替换为您的文件路径

# 1. 数据预处理：按日期汇总
# 将日期列转换为 datetime 类型
df['日期'] = pd.to_datetime(df['日期'])

# 按日期对【总收入_税后】求和
df_grouped = df.groupby('日期', as_index=False).agg({'总收入_税后': 'sum'})

# 获取数值型辅助字段（排除包含非数值数据的列）
numeric_columns = [col for col in df.columns if col not in ['日期', '总收入_税后', '市场类型名称'] and pd.to_numeric(df[col], errors='coerce').notna().all()]

# 对其他数值型辅助字段按日期取平均值
df_auxiliary = df.groupby('日期', as_index=False)[numeric_columns].mean()

# 将汇总的辅助字段数据与汇总后的【总收入_税后】合并
df_grouped = df_grouped.merge(df_auxiliary, on='日期', how='left')

# 2. 准备Prophet模型输入数据
df_grouped = df_grouped.rename(columns={'日期': 'ds', '总收入_税后': 'y'})

# 3. 拆分训练集和测试集
train_size = int(len(df_grouped) * 0.8)
train = df_grouped[:train_size]
test = df_grouped[train_size:]

# 4. 初始化并配置Prophet模型
model = Prophet()
for col in numeric_columns:
    model.add_regressor(col)

# 5. 训练模型
model.fit(train)

# 6. 生成未来日期并添加辅助字段
future = model.make_future_dataframe(periods=len(test))

# 确保未来数据的日期格式一致
future['ds'] = pd.to_datetime(future['ds'])
for col in numeric_columns:
    future = future.merge(df_grouped[['ds', col]].drop_duplicates(), on='ds', how='left')

# 7. 预测
forecast = model.predict(future)

# 8. 计算R方
y_true = test['y'].values
y_pred = forecast.loc[forecast['ds'].isin(test['ds']), 'yhat'].values
r2 = r2_score(y_true, y_pred)

# 9. 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(train['ds'], train['y'], label='Train', color='blue')
plt.plot(test['ds'], test['y'], label='Test', color='orange')
plt.plot(forecast['ds'], forecast['yhat'], label='Prediction', color='green')
plt.xlabel('Date')
plt.ylabel('总收入_税后')
plt.title(f'Time Series Forecasting (R-squared: {r2:.2f})')
plt.legend()
plt.show()
