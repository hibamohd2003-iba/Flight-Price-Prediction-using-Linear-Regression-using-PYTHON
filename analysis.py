# ============================================================
# Flight Price Prediction - Linear Regression in Python
# Author: Hiba Muhammed | Roll No: 25 | S6IE
# ============================================================

# ---- 1. Import Libraries ----
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="Set2")
print("Libraries loaded successfully.")

# ============================================================
# 2. Load & Explore Dataset
# ============================================================
df = pd.read_csv("flight-data.csv", index_col=0)

print("\n--- Dataset Overview ---")
print(f"Shape        : {df.shape}")
print(f"Columns      : {list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nBasic Statistics:\n{df.describe()}")
print(f"\nUnique values per column:")
for col in df.columns:
    print(f"  {col}: {df[col].nunique()} unique → {df[col].unique()[:5]}")

# ============================================================
# 3. EDA
# ============================================================

# ---- Q1: Does price vary with Airlines (same route)? ----
print("\n--- Q1: Price by Airline (Delhi → Mumbai) ---")
route = df[(df['source_city'] == 'Delhi') & (df['destination_city'] == 'Mumbai')]
print(route.groupby('airline')['price'].describe().round(2))

fig, ax = plt.subplots(figsize=(9, 5))
order = route.groupby('airline')['price'].median().sort_values().index
sns.boxplot(data=route, x='airline', y='price', order=order, palette='Set2',
            flierprops=dict(marker='.', alpha=0.3, markersize=2), ax=ax)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
ax.set_title('Flight Price by Airline (Delhi → Mumbai)', fontsize=14, fontweight='bold')
ax.set_xlabel('Airline')
ax.set_ylabel('Ticket Price (INR)')
plt.tight_layout()
plt.savefig('plot1.png', dpi=150)
plt.close()
print("plot1.png saved")

# ---- Q2: Price when bought 1-2 days before departure ----
print("\n--- Q2: Price vs Days Before Departure ---")

def booking_group(d):
    if d <= 2:   return '1-2 Days\nBefore'
    elif d <= 7:  return '3-7 Days\nBefore'
    elif d <= 14: return '8-14 Days\nBefore'
    elif d <= 30: return '15-30 Days\nBefore'
    else:         return '30+ Days\nBefore'

df['booking_group'] = df['days_left'].apply(booking_group)
order2 = ['1-2 Days\nBefore', '3-7 Days\nBefore', '8-14 Days\nBefore',
          '15-30 Days\nBefore', '30+ Days\nBefore']

print(df.groupby('booking_group')['price'].median().reindex(order2))

fig, ax = plt.subplots(figsize=(9, 5))
palette = sns.color_palette("RdYlGn", 5)[::-1]
sns.boxplot(data=df, x='booking_group', y='price', order=order2, palette=palette,
            flierprops=dict(marker='.', alpha=0.2, markersize=1.5), ax=ax)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
ax.set_title('Ticket Price vs. Days Before Departure', fontsize=14, fontweight='bold')
ax.set_xlabel('Booking Window')
ax.set_ylabel('Ticket Price (INR)')
plt.tight_layout()
plt.savefig('plot2.png', dpi=150)
plt.close()
print("plot2.png saved")

# ---- Q3: Price based on departure & arrival time ----
print("\n--- Q3: Price by Departure & Arrival Time ---")
time_order = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night']
pivot = df.groupby(['departure_time', 'arrival_time'])['price'].mean().unstack()
pivot = pivot.reindex(index=time_order, columns=time_order)
print(pivot.round(0))

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd',
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Avg Price (INR)'})
ax.set_title('Average Price by Departure & Arrival Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Arrival Time')
ax.set_ylabel('Departure Time')
plt.tight_layout()
plt.savefig('plot3.png', dpi=150)
plt.close()
print("plot3.png saved")

# ============================================================
# 4. Price Distribution & Outlier Removal (IQR)
# ============================================================
print("\n--- Price Distribution & Outlier Removal ---")
print(f"Price range before: ₹{df['price'].min():,} – ₹{df['price'].max():,}")
print(f"Mean: ₹{df['price'].mean():,.2f}  |  Std: ₹{df['price'].std():,.2f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df['price'], bins=80, color='#4575b4', alpha=0.85, edgecolor='white')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
axes[0].set_title('Price Distribution\n(Before Outlier Removal)', fontweight='bold')
axes[0].set_xlabel('Price (INR)')
axes[0].set_ylabel('Frequency')

# IQR Method
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"\nQ1={Q1:.0f}, Q3={Q3:.0f}, IQR={IQR:.0f}")
print(f"Lower bound: {lower_bound:.0f} | Upper bound: {upper_bound:.0f}")

df_clean = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)].copy()
print(f"Rows before: {len(df):,} | After: {len(df_clean):,} | Removed: {len(df)-len(df_clean)}")

axes[1].hist(df_clean['price'], bins=80, color='#1a9641', alpha=0.85, edgecolor='white')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
axes[1].set_title('Price Distribution\n(After Outlier Removal)', fontweight='bold')
axes[1].set_xlabel('Price (INR)')
axes[1].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('plot4.png', dpi=150)
plt.close()
print("plot4.png saved")

# ============================================================
# 5. Feature Engineering & Preprocessing
# ============================================================
print("\n--- Feature Engineering ---")

df_model = df_clean.drop(columns=['flight', 'booking_group'])

cat_cols = ['airline', 'source_city', 'departure_time', 'stops',
            'arrival_time', 'destination_city', 'class']
num_cols = ['duration', 'days_left']

ct = ColumnTransformer(
    [('ohe', OneHotEncoder(drop='first', sparse_output=False), cat_cols)],
    remainder='passthrough'
)

X = df_model.drop('price', axis=1)
y = df_model['price']

X_enc = ct.fit_transform(X)
feat_names = ct.get_feature_names_out()
print(f"Total features after encoding: {X_enc.shape[1]}")
print(f"Feature names: {list(feat_names)}")

# ============================================================
# 6. Train/Test Split (80:20)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")

# ============================================================
# 7. Model M1 — Linear Regression (All Features) — Scikit-Learn
# ============================================================
print("\n--- Model M1: Linear Regression (All Features) via Scikit-Learn ---")
m1 = LinearRegression()
m1.fit(X_train, y_train)
y_pred_m1 = m1.predict(X_test)

m1_r2   = r2_score(y_test, y_pred_m1)
m1_rmse = np.sqrt(mean_squared_error(y_test, y_pred_m1))
print(f"R²   : {m1_r2:.4f}")
print(f"RMSE : ₹{m1_rmse:,.2f}")

# ============================================================
# 8. OLS Regression (Statsmodels) — M1
# ============================================================
print("\n--- OLS Regression Summary (Statsmodels) - M1 ---")
X_train_sm = sm.add_constant(X_train)
ols_m1 = sm.OLS(y_train, X_train_sm).fit()
print(ols_m1.summary())

# ============================================================
# 9. Feature Selection — Top 5 by Coefficient Magnitude
# ============================================================
coef_abs = np.abs(m1.coef_)
top5_idx   = np.argsort(coef_abs)[::-1][:5]
top5_names = [feat_names[i] for i in top5_idx]
print(f"\nTop 5 Features by Coefficient Magnitude:")
for i, (name, coef) in enumerate(zip(top5_names, m1.coef_[top5_idx]), 1):
    print(f"  {i}. {name.replace('ohe__','')}: {coef:.4f}")

# ============================================================
# 10. Model M2 — Linear Regression (Top 5 Features) — Scikit-Learn
# ============================================================
print("\n--- Model M2: Linear Regression (Top 5 Features) via Scikit-Learn ---")
X_train_5 = X_train[:, top5_idx]
X_test_5  = X_test[:, top5_idx]

m2 = LinearRegression()
m2.fit(X_train_5, y_train)
y_pred_m2 = m2.predict(X_test_5)

m2_r2   = r2_score(y_test, y_pred_m2)
m2_rmse = np.sqrt(mean_squared_error(y_test, y_pred_m2))
print(f"R²   : {m2_r2:.4f}")
print(f"RMSE : ₹{m2_rmse:,.2f}")

# ============================================================
# 11. OLS Regression (Statsmodels) — M2
# ============================================================
print("\n--- OLS Regression Summary (Statsmodels) - M2 ---")
X_train_5_sm = sm.add_constant(X_train_5)
ols_m2 = sm.OLS(y_train, X_train_5_sm).fit()
print(ols_m2.summary())

# ============================================================
# 12. Model Comparison
# ============================================================
print("\n" + "="*55)
print("         MODEL COMPARISON: M1 vs M2")
print("="*55)
print(f"{'Metric':<20} {'M1 (All Features)':<22} {'M2 (Top 5)'}")
print("-"*55)
print(f"{'R²':<20} {m1_r2:<22.4f} {m2_r2:.4f}")
print(f"{'Adj R² (OLS)':<20} {ols_m1.rsquared_adj:<22.4f} {ols_m2.rsquared_adj:.4f}")
print(f"{'RMSE':<20} ₹{m1_rmse:<21,.2f} ₹{m2_rmse:,.2f}")
print("="*55)

# ============================================================
# 13. Actual vs Predicted Plot (M1)
# ============================================================
idx = np.random.choice(len(y_test), 5000, replace=False)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(np.array(y_test)[idx], y_pred_m1[idx],
           alpha=0.3, s=5, color='#4575b4')
lims = [0, max(y_test.max(), y_pred_m1.max())]
ax.plot(lims, lims, 'r-', lw=1.5, label='Perfect Fit')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
ax.set_title(f'Actual vs Predicted Price (M1 – All Features)\nR² = {m1_r2:.4f} | RMSE = ₹{m1_rmse:,.0f}',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Actual Price (INR)')
ax.set_ylabel('Predicted Price (INR)')
ax.legend()
plt.tight_layout()
plt.savefig('plot5.png', dpi=150)
plt.close()
print("\nplot5.png saved")

# ============================================================
# 14. Feature Importance Plot
# ============================================================
top10_idx   = np.argsort(coef_abs)[::-1][:10]
top10_names = [feat_names[i].replace('ohe__', '').replace('remainder__', '')
               for i in top10_idx]
top10_coefs = m1.coef_[top10_idx]

colors_bar = ['#1a9641' if c > 0 else '#d73027' for c in top10_coefs]
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(range(10), top10_coefs[::-1], color=colors_bar[::-1], alpha=0.85)
ax.set_yticks(range(10))
ax.set_yticklabels(top10_names[::-1], fontsize=10)
ax.set_title('Top 10 Features by Coefficient Magnitude (M1)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Coefficient Value')
ax.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('plot6.png', dpi=150)
plt.close()
print("plot6.png saved")

# ============================================================
print("\n" + "="*55)
print("  Analysis Complete! All plots and models saved.")
print("="*55)
