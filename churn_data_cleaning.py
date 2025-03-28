# Importing the required packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# Load datasets
client_data = pd.read_csv("client_data (1).csv")
price_data = pd.read_csv("price_data (1).csv")

# ----------------------------------------------------------------
# Display top 5 rows
print(client_data.head())
print(price_data.head())

# ----------------------------------------------------------------
# Descriptive statistics
print(client_data.describe())
print(price_data.describe())

# ----------------------------------------------------------------
# Info about datasets
print(client_data.info())
print(price_data.info())

# ----------------------------------------------------------------
# Columns in datasets
print("Client data columns:", client_data.columns.tolist())
print("Price data columns:", price_data.columns.tolist())

# ----------------------------------------------------------------
# Stacked Bar Chart: Consumption and Margin gross by churn
features_to_include = ['cons_12m', 'cons_gas_12m', 'margin_gross_pow_ele']
filtered_data = client_data[['churn'] + features_to_include]
churn_counts = filtered_data.groupby('churn').sum()

ax = churn_counts.plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Selected Features by Churn')
plt.xlabel('Churn')
plt.ylabel('Sum of Feature Values')
plt.legend(title='Features')
plt.show()

# ----------------------------------------------------------------
# Forecasting features by churn
forecast_features = [
    'forecast_cons_12m',
    'forecast_cons_year',
    'forecast_discount_energy',
    'forecast_meter_rent_12m',
    'forecast_price_energy_off_peak',
    'forecast_price_pow_off_peak'
]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(forecast_features):
    sns.histplot(data=client_data, x=col, hue='churn', kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col} by Churn')
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------
# More features by churn
features = ['cons_12m', 'cons_gas_12m', 'net_margin', 'num_years_antig', 'pow_max']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(features):
    sns.histplot(data=client_data, x=col, hue='churn', kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col} by Churn')
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------
# Merge client and price data
merge_data = pd.merge(client_data, price_data, on='id', how='inner')
print(merge_data.head())

# ----------------------------------------------------------------
# Boxplot: Price off/mid with churn
features_of_price = ['price_off_peak_var', 'price_mid_peak_var']
for feature in features_of_price:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='churn', y=feature, data=merge_data)
    plt.title('Visualizing the price with churn')
    plt.xlabel('Churn')
    plt.ylabel(feature)
    plt.show()

# ----------------------------------------------------------------
# Violin plot: price_mid_peak_var vs churn
sns.violinplot(x='churn', y='price_mid_peak_var', data=merge_data)
plt.title("Visualizing the price with churn")
plt.show()
