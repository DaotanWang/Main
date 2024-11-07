
import pandas as pd
from prophet import Prophet
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df = pd.read_excel('D:\\user\\Downloads\\all_.xlsx')  # 请替换为您的文件路径

df['日期'] = pd.to_datetime(df['日期'])

df_grouped = df.groupby('日期', as_index=False).agg({'总收入_税后': 'sum'})

numeric_columns = [col for col in df.columns if col not in ['日期', '总收入_税后', '市场类型名称'] and pd.to_numeric(df[col], errors='coerce').notna().all()]

df_auxiliary = df.groupby('日期', as_index=False)[numeric_columns].mean()

df_grouped = df_grouped.merge(df_auxiliary, on='日期', how='left')

df_grouped = df_grouped.rename(columns={'日期': 'ds', '总收入_税后': 'y'})

train_size = int(len(df_grouped) * 0.8)
train = df_grouped[:train_size]
test = df_grouped[train_size:]

model = Prophet()
for col in numeric_columns:
    model.add_regressor(col)

model.fit(train)

future = model.make_future_dataframe(periods=len(test))

future['ds'] = pd.to_datetime(future['ds'])
for col in numeric_columns:
    future = future.merge(df_grouped[['ds', col]].drop_duplicates(), on='ds', how='left')

forecast = model.predict(future)

y_true = test['y'].values
y_pred = forecast.loc[forecast['ds'].isin(test['ds']), 'yhat'].values
r2 = r2_score(y_true, y_pred)

plt.figure(figsize=(10, 6))
plt.plot(train['ds'], train['y'], label='Train', color='blue')
plt.plot(test['ds'], test['y'], label='Test', color='orange')
plt.plot(forecast['ds'], forecast['yhat'], label='Prediction', color='green')
plt.xlabel('Date')
plt.ylabel('总收入_税后')
plt.title(f'Time Series Forecasting (R-squared: {r2:.2f})')
plt.legend()
plt.show()
