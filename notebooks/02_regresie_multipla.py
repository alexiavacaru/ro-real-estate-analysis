import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 120
sns.set_theme(style='whitegrid')

df = pd.read_csv('data/processed/dataset_final.csv')
print(df.shape)
print(df.head())

# 1. Pregătire date — encoding variabile categoriale
le_oras = LabelEncoder()
le_regiune = LabelEncoder()
le_tip = LabelEncoder()
le_zona = LabelEncoder()

df['oras_enc'] = le_oras.fit_transform(df['oras'])
df['regiune_enc'] = le_regiune.fit_transform(df['regiune'])
df['tip_enc'] = le_tip.fit_transform(df['tip_prop'])
df['zona_enc'] = le_zona.fit_transform(df['zona'])

df['vechime'] = 2024 - df['an_constructie']

features = ['suprafata_mp', 'camere', 'vechime', 'oras_enc', 'regiune_enc',
            'HPI_trim', 'IRCC_trim_pct', 'autorizatii_judet_an']
target = 'pret_eur_mp'

X = df[features].copy()
y = df[target].copy()

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# 2. Split train/test (80/20) — seed 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape[0]} obs | Test: {X_test.shape[0]} obs")

# 3. Model 1 — Regresie Liniară Simplă (OLS)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

ols = LinearRegression()
ols.fit(X_train_sc, y_train)

y_pred_ols = ols.predict(X_test_sc)

r2_ols = r2_score(y_test, y_pred_ols)
rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))
mae_ols = mean_absolute_error(y_test, y_pred_ols)

print(f"OLS — R²: {r2_ols:.4f} | RMSE: {rmse_ols:.2f} | MAE: {mae_ols:.2f}")

# 4. Model 2 — Ridge Regression (L2)
from sklearn.linear_model import RidgeCV

alphas = [0.01, 0.1, 1, 10, 100, 500]
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='r2')
ridge_cv.fit(X_train_sc, y_train)

print(f"Ridge — Alpha optim: {ridge_cv.alpha_}")

y_pred_ridge = ridge_cv.predict(X_test_sc)
r2_ridge = r2_score(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

print(f"Ridge — R²: {r2_ridge:.4f} | RMSE: {rmse_ridge:.2f} | MAE: {mae_ridge:.2f}")

# 5. Model 3 — Lasso Regression (L1)
from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=5000, random_state=42)
lasso_cv.fit(X_train_sc, y_train)

print(f"Lasso — Alpha optim: {lasso_cv.alpha_:.4f}")

y_pred_lasso = lasso_cv.predict(X_test_sc)
r2_lasso = r2_score(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

print(f"Lasso — R²: {r2_lasso:.4f} | RMSE: {rmse_lasso:.2f} | MAE: {mae_lasso:.2f}")

# 6. Cross-Validation 10-fold pe modelul OLS
kf = KFold(n_splits=10, shuffle=True, random_state=42)

pipe = Pipeline([('scaler', StandardScaler()), ('ols', LinearRegression())])
cv_scores = cross_val_score(pipe, X, y, cv=kf, scoring='r2')

print(f"CV R² scores: {np.round(cv_scores, 4)}")
print(f"Media R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 7. Comparație modele
rezultate = pd.DataFrame({
    'Model': ['OLS', 'Ridge', 'Lasso'],
    'R²': [r2_ols, r2_ridge, r2_lasso],
    'RMSE (€/mp)': [rmse_ols, rmse_ridge, rmse_lasso],
    'MAE (€/mp)': [mae_ols, mae_ridge, mae_lasso]
})
print(rezultate.round(4).to_string(index=False))

# 8. Coeficienți OLS — importanța variabilelor
coef_df = pd.DataFrame({
    'Feature': features,
    'Coeficient': ols.coef_
}).sort_values('Coeficient', key=abs, ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#C00000' if c < 0 else '#1F4E79' for c in coef_df['Coeficient']]
ax.barh(coef_df['Feature'], coef_df['Coeficient'], color=colors, alpha=0.85)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Coeficienți OLS standardizați — importanța variabilelor', fontsize=12, fontweight='bold')
ax.set_xlabel('Coeficient (variabile standardizate)')
plt.tight_layout()
plt.savefig('outputs/figures/02_coeficienti_ols.png', bbox_inches='tight')
plt.show()

# 9. Grafice diagnostic — Residuri
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

reziduri = y_test - y_pred_ols

axes[0].scatter(y_pred_ols, reziduri, alpha=0.7, color='#2E75B6', edgecolors='white', s=70)
axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[0].set_xlabel('Valori ajustate (€/mp)')
axes[0].set_ylabel('Reziduuri')
axes[0].set_title('Reziduuri vs. Valori ajustate')

axes[1].hist(reziduri, bins=12, color='#2E75B6', alpha=0.8, edgecolor='white')
axes[1].set_xlabel('Reziduuri (€/mp)')
axes[1].set_ylabel('Frecvență')
axes[1].set_title('Distribuția reziduurilor')

axes[2].scatter(y_test, y_pred_ols, alpha=0.75, color='#1F4E79', edgecolors='white', s=70)
lim = [min(y_test.min(), y_pred_ols.min()) - 50, max(y_test.max(), y_pred_ols.max()) + 50]
axes[2].plot(lim, lim, 'r--', linewidth=2, label='Predicție perfectă')
axes[2].set_xlabel('Valori reale (€/mp)')
axes[2].set_ylabel('Valori prezise (€/mp)')
axes[2].set_title(f'Real vs. Prezis (R²={r2_ols:.3f})')
axes[2].legend()

plt.tight_layout()
plt.savefig('outputs/figures/02_diagnostic_reziduri.png', bbox_inches='tight')
plt.show()

# 10. Lasso — selectie variabile (coeficienți nenuli)
lasso_coef = pd.DataFrame({
    'Feature': features,
    'Coeficient Lasso': lasso_cv.coef_
})
print("Variabile selectate de Lasso (coef != 0):")
print(lasso_coef[lasso_coef['Coeficient Lasso'] != 0].to_string(index=False))
print()
print("Variabile eliminate (coef = 0):")
print(lasso_coef[lasso_coef['Coeficient Lasso'] == 0]['Feature'].tolist())
