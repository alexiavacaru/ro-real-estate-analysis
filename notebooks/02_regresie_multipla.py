
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.linear_model   import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing  import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics        import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline       import Pipeline

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 120
sns.set_theme(style='whitegrid')

print("Gata de lucru.")

df = pd.read_csv('dataset_final.csv')

print(f"Shape dataset: {df.shape}")
print("\nColoane disponibile:")
print(df.columns.tolist())
display(df.head())


le_oras    = LabelEncoder()
le_regiune = LabelEncoder()
le_tip     = LabelEncoder()
le_zona    = LabelEncoder()

df['oras_enc']    = le_oras.fit_transform(df['oras'])
df['regiune_enc'] = le_regiune.fit_transform(df['regiune'])
df['tip_enc']     = le_tip.fit_transform(df['tip_prop'])
df['zona_enc']    = le_zona.fit_transform(df['zona'])

# Calculăm vechimea construcției față de 2024
df['vechime'] = 2024 - df['an_constructie']

# Afișăm maparea orașelor ca să știm ce înseamnă fiecare cod
print("Mapping oras → cod numeric:")
for idx, oras in enumerate(le_oras.classes_):
    print(f"  {idx} → {oras}")


features = [
    'suprafata_mp',
    'camere',
    'vechime',
    'oras_enc',
    'regiune_enc',
    'HPI_trim',
    'IRCC_trim_pct',
    'autorizatii_judet_an'
]

target = 'pret_eur_mp'

X = df[features].copy()
y = df[target].copy()

print(f"Features: {X.shape[1]} variabile, {X.shape[0]} observații")
print(f"Target: {target}")
print(f"\nStatistici target:")
print(y.describe().round(1))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Set antrenare: {X_train.shape[0]} observații")
print(f"Set testare:   {X_test.shape[0]} observații")

# Standardizăm datele — important pentru Ridge și Lasso
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

ols = LinearRegression()
ols.fit(X_train_sc, y_train)
y_pred_ols = ols.predict(X_test_sc)

r2_ols   = r2_score(y_test, y_pred_ols)
rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))
mae_ols  = mean_absolute_error(y_test, y_pred_ols)

print("=== Model OLS (Regresie Liniară) ===")
print(f"R²   : {r2_ols:.4f}")
print(f"RMSE : {rmse_ols:.2f} €/mp")
print(f"MAE  : {mae_ols:.2f} €/mp")

alphas = [0.01, 0.1, 1, 10, 100, 500]
ridge  = RidgeCV(alphas=alphas, cv=5, scoring='r2')
ridge.fit(X_train_sc, y_train)

y_pred_ridge = ridge.predict(X_test_sc)
r2_ridge     = r2_score(y_test, y_pred_ridge)
rmse_ridge   = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
mae_ridge    = mean_absolute_error(y_test, y_pred_ridge)

print("=== Model Ridge (L2 regularizare) ===")
print(f"Alpha optim ales prin CV: {ridge.alpha_}")
print(f"R²   : {r2_ridge:.4f}")
print(f"RMSE : {rmse_ridge:.2f} €/mp")
print(f"MAE  : {mae_ridge:.2f} €/mp")


lasso = LassoCV(alphas=alphas, cv=5, max_iter=5000, random_state=42)
lasso.fit(X_train_sc, y_train)

y_pred_lasso = lasso.predict(X_test_sc)
r2_lasso     = r2_score(y_test, y_pred_lasso)
rmse_lasso   = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
mae_lasso    = mean_absolute_error(y_test, y_pred_lasso)

print("=== Model Lasso (L1 regularizare + selecție variabile) ===")
print(f"Alpha optim ales prin CV: {lasso.alpha_:.4f}")
print(f"R²   : {r2_lasso:.4f}")
print(f"RMSE : {rmse_lasso:.2f} €/mp")
print(f"MAE  : {mae_lasso:.2f} €/mp")


kf = KFold(n_splits=10, shuffle=True, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('ols',    LinearRegression())
])

cv_scores = cross_val_score(pipe, X, y, cv=kf, scoring='r2')

print("=== Cross-Validation 10-fold (OLS) ===")
print(f"R² per fold: {np.round(cv_scores, 4)}")
print(f"Media R²:    {cv_scores.mean():.4f}")
print(f"Std R²:      {cv_scores.std():.4f}")

rezultate = pd.DataFrame({
    'Model':         ['OLS', 'Ridge', 'Lasso'],
    'R²':            [r2_ols,    r2_ridge,    r2_lasso],
    'RMSE (€/mp)':   [rmse_ols,  rmse_ridge,  rmse_lasso],
    'MAE (€/mp)':    [mae_ols,   mae_ridge,   mae_lasso],
})

print("\n=== Comparație finală modele ===")
display(rezultate.round(4))


coef_df = (pd.DataFrame({'Feature': features, 'Coeficient': ols.coef_})
             .sort_values('Coeficient', key=abs, ascending=True))

fig, ax = plt.subplots(figsize=(9, 5))
culori_coef = ['#C00000' if c < 0 else '#1F4E79' for c in coef_df['Coeficient']]
ax.barh(coef_df['Feature'], coef_df['Coeficient'], color=culori_coef, alpha=0.85)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Coeficienți OLS standardizați — importanța variabilelor',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Coeficient (variabile standardizate)')
plt.tight_layout()
plt.savefig('02_coeficienti_ols.png', bbox_inches='tight')
plt.show()


reziduri = y_test - y_pred_ols

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Reziduuri vs. valori ajustate
axes[0].scatter(y_pred_ols, reziduri, alpha=0.7,
                color='#2E75B6', edgecolors='white', s=70)
axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[0].set_xlabel('Valori ajustate (€/mp)')
axes[0].set_ylabel('Reziduuri')
axes[0].set_title('Reziduuri vs. Valori ajustate')

# Distribuția reziduurilor
axes[1].hist(reziduri, bins=12, color='#2E75B6', alpha=0.8, edgecolor='white')
axes[1].set_xlabel('Reziduuri (€/mp)')
axes[1].set_ylabel('Frecvență')
axes[1].set_title('Distribuția reziduurilor')

# Real vs. Prezis
lim_min = min(y_test.min(), y_pred_ols.min()) - 50
lim_max = max(y_test.max(), y_pred_ols.max()) + 50
axes[2].scatter(y_test, y_pred_ols, alpha=0.75,
                color='#1F4E79', edgecolors='white', s=70)
axes[2].plot([lim_min, lim_max], [lim_min, lim_max],
             'r--', linewidth=2, label='Predicție perfectă')
axes[2].set_xlabel('Valori reale (€/mp)')
axes[2].set_ylabel('Valori prezise (€/mp)')
axes[2].set_title(f'Real vs. Prezis (R²={r2_ols:.3f})')
axes[2].legend()

plt.tight_layout()
plt.savefig('02_diagnostic_reziduri.png', bbox_inches='tight')
plt.show()

lasso_coef = pd.DataFrame({
    'Feature':           features,
    'Coeficient Lasso':  lasso.coef_
})

print("Variabile PĂSTRATE de Lasso (coeficient ≠ 0):")
display(lasso_coef[lasso_coef['Coeficient Lasso'] != 0])

print("\nVariabile ELIMINATE de Lasso (coeficient = 0):")
eliminate = lasso_coef[lasso_coef['Coeficient Lasso'] == 0]['Feature'].tolist()
print(eliminate if eliminate else "Nicio variabilă eliminată")
