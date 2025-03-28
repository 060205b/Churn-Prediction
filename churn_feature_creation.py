# feature_engineering_script.py

# ....................................................................................
# 1. Import Libraries
# ....................................................................................
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ....................................................................................
# 2. Load Dataset
# ....................................................................................
clean_dataset = pd.read_csv("clean_data_after_eda.csv")
print("✅ Loaded clean dataset with shape:", clean_dataset.shape)

# ....................................................................................
# 3. Preview Dataset
# ....................................................................................
print("\n🔹 Sample Rows:\n", clean_dataset.head())
print("\n🔹 Columns:\n", clean_dataset.columns.tolist())

# ....................................................................................
# 4. Feature Engineering - Active Time
# ....................................................................................
clean_dataset['days_active'] = (pd.to_datetime('today') - pd.to_datetime(clean_dataset['date_activ'])).dt.days
clean_dataset['months_active'] = clean_dataset['days_active'] / 30
clean_dataset['years_active'] = clean_dataset['days_active'] / 365

# ....................................................................................
# 5. Ratios
# ....................................................................................
clean_dataset['monthly_to_yearly_ratio'] = clean_dataset['cons_last_month'] / clean_dataset['cons_12m']
clean_dataset['forecast_to_actual_ratio'] = clean_dataset['forecast_cons_12m'] / clean_dataset['cons_12m']

print("\n🔹 Ratio Columns:\n", clean_dataset[['monthly_to_yearly_ratio', 'forecast_to_actual_ratio']].head())

# ....................................................................................
# 6. Price Differences and Margin Calculations
# ....................................................................................
clean_dataset['price_peak_offpeak_diff'] = clean_dataset['forecast_price_energy_peak'] - clean_dataset['forecast_price_energy_off_peak']
clean_dataset['fixed_vs_var_peak'] = clean_dataset['var_year_price_peak_fix'] - clean_dataset['var_year_price_peak_var']
print("\n🔹 Price Diffs:\n", clean_dataset[['price_peak_offpeak_diff', 'fixed_vs_var_peak']].head())

clean_dataset['profit_margin_per_prod'] = clean_dataset['net_margin'] / clean_dataset['nb_prod_act']
clean_dataset['gross_vs_net_margin_ratio'] = clean_dataset['margin_gross_pow_ele'] / clean_dataset['margin_net_pow_ele']
print("\n🔹 Margins:\n", clean_dataset[['profit_margin_per_prod', 'gross_vs_net_margin_ratio']].head())

# ....................................................................................
# 7. Binning Features
# ....................................................................................
clean_dataset['pow_max_bin'] = pd.cut(clean_dataset['pow_max'], bins=[0, 10, 20, 30, 100], labels=['low', 'medium', 'high', 'very_high'])
clean_dataset['tenure_bin'] = pd.cut(clean_dataset['num_years_antig'], bins=[0, 1, 3, 10], labels=['new', 'intermediate', 'long-term'])

# ....................................................................................
# 8. Fix has_gas Column
# ....................................................................................
clean_dataset['has_gas'] = clean_dataset['has_gas'].fillna(0)
clean_dataset['has_gas'] = clean_dataset['has_gas'].replace(['t', 'f'], [1, 0])
print("\n🔹 Churn rate by Gas:\n", clean_dataset.groupby('has_gas')['churn'].mean())

# ....................................................................................
# 9. Load Price Data
# ....................................................................................
price_data = pd.read_csv("price_data (1).csv")

# ....................................................................................
# 10. Monthly Price Difference (Dec vs Jan)
# ....................................................................................
monthly_price_by_id = price_data.groupby(['id', 'price_date']).agg({
    'price_off_peak_var': 'mean',
    'price_off_peak_fix': 'mean'
}).reset_index()

jan_prices = monthly_price_by_id.groupby('id').first().reset_index()
dec_prices = monthly_price_by_id.groupby('id').last().reset_index()

diff = pd.merge(
    dec_prices.rename(columns={'price_off_peak_var': 'dec_1', 'price_off_peak_fix': 'dec_2'}),
    jan_prices.drop(columns='price_date'),
    on='id'
)
diff['offpeak_diff_dec_january_energy'] = diff['dec_1'] - diff['price_off_peak_var']
diff['offpeak_diff_dec_january_power'] = diff['dec_2'] - diff['price_off_peak_fix']

print("\n🔹 Monthly Price Diff:\n", diff[['offpeak_diff_dec_january_energy', 'offpeak_diff_dec_january_power']].head())

# ....................................................................................
# 11. Descriptive Statistics of Consumption
# ....................................................................................
cols = ['cons_12m', 'cons_gas_12m', 'cons_last_month','forecast_cons_12m',
        'forecast_cons_year', 'forecast_discount_energy','forecast_meter_rent_12m',
        'forecast_price_energy_off_peak','forecast_price_energy_peak', 'forecast_price_pow_off_peak']

print("\n🔹 Consumption & Forecast Stats:\n", clean_dataset[cols].describe())

# ....................................................................................
# 12. Merge Clean Data + Price Data
# ....................................................................................
merge_price_clean_data = pd.merge(price_data, clean_dataset, on='id', how='inner')

# ....................................................................................
# 13. Additional Features
# ....................................................................................
merge_price_clean_data['price_elasticity'] = (merge_price_clean_data['forecast_cons_12m'] - merge_price_clean_data['cons_12m']) / merge_price_clean_data['forecast_price_energy_peak']
merge_price_clean_data['price_increase'] = merge_price_clean_data['var_year_price_peak_var'] - merge_price_clean_data['var_6m_price_peak_var']
merge_price_clean_data['price_increase_churn'] = (merge_price_clean_data['price_increase'] > 0.10).astype(int)

merge_price_clean_data['margin_vs_peak_price'] = merge_price_clean_data['margin_gross_pow_ele'] / merge_price_clean_data['forecast_price_energy_peak']
merge_price_clean_data['price_trend_6m_to_year'] = merge_price_clean_data['var_6m_price_peak_var'] / merge_price_clean_data['var_year_price_peak_var']
merge_price_clean_data['price_stability'] = merge_price_clean_data[['var_year_price_peak_var', 'var_6m_price_peak_var']].std(axis=1)
merge_price_clean_data['revenue_per_customer'] = merge_price_clean_data['cons_12m'] * merge_price_clean_data['forecast_price_energy_peak']

# ....................................................................................
# 14. Correlation Matrix Heatmap
# ....................................................................................
numeric_columns = clean_dataset.select_dtypes(include=['number'])
correlation_matrix = numeric_columns.corr().fillna(0)

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, annot_kws={'size': 3})
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()
