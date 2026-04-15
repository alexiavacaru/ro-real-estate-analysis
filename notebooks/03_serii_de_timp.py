import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 120
sns.set_theme(style='whitegrid')

df = pd.read_csv('data/processed/serii_timp_pret_mp.csv')
print(df.head())

buc = df[df['oras'] == 'Bucuresti'].copy().reset_index(drop=True)
clj = df[df['oras'] == 'Cluj-Napoca'].copy().reset_index(drop=True)
tim = df[df['oras'] == 'Timisoara'].copy().reset_index(drop=True)

buc['t'] = range(len(buc))
clj['t'] = range(len(clj))
tim['t'] = range(len(tim))

buc['data'] = pd.to_datetime(buc['an'].astype(str) + '-' + 
                              buc['trimestru'].str.replace('T1','01').str.replace('T2','04')
                              .str.replace('T3','07').str.replace('T4','10'))
clj['data'] = pd.to_datetime(clj['an'].astype(str) + '-' + 
                              clj['trimestru'].str.replace('T1','01').str.replace('T2','04')
                              .str.replace('T3','07').str.replace('T4','10'))

print(f"București: {len(buc)} obs | Cluj: {len(clj)} obs | Timișoara: {len(tim)} obs")

# 1. Vizualizare serii brute
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(buc['t'], buc['pret_mediu_eur_mp'], 'o-', color='#1F4E79', linewidth=2.5, 
        markersize=5, label='București')
ax.plot(clj['t'], clj['pret_mediu_eur_mp'], 's-', color='#C00000', linewidth=2.5, 
        markersize=5, label='Cluj-Napoca')
ax.plot(tim['t'], tim['pret_mediu_eur_mp'], '^-', color='#2E75B6', linewidth=2.5, 
        markersize=5, label='Timișoara')

tick_labels = [f"{r['trimestru']}\n{r['an']}" for _, r in buc.iterrows()]
ax.set_xticks(buc['t'])
ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)
ax.set_title('Evoluția prețului mediu €/mp — Serii trimestriale 2019–2024', 
             fontsize=13, fontweight='bold')
ax.set_ylabel('Preț mediu (€/mp)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('outputs/figures/03_serii_brute.png', bbox_inches='tight')
plt.show()

# 2. Descompunere manuală: Trend (medie mobilă) + Sezonalitate
def moving_average(series, window=4):
    return pd.Series(series).rolling(window=window, center=True).mean().values

buc_ma = moving_average(buc['pret_mediu_eur_mp'])
buc_trend_desezon = buc['pret_mediu_eur_mp'].values - np.nan_to_num(buc_ma, nan=buc['pret_mediu_eur_mp'].mean())

trim_means = {}
for t in ['T1','T2','T3','T4']:
    vals = buc[buc['trimestru'] == t]['pret_mediu_eur_mp'].values
    trim_means[t] = np.mean(vals) - buc['pret_mediu_eur_mp'].mean()

sezon = buc['trimestru'].map(trim_means).values

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(buc['t'], buc['pret_mediu_eur_mp'], color='#1F4E79', linewidth=2, label='Serie originală')
axes[0].plot(buc['t'], buc_ma, color='#C00000', linewidth=2.5, linestyle='--', label='Trend (MA-4)')
axes[0].set_title('Serie originală + Trend (medie mobilă 4 trimestre)', fontweight='bold')
axes[0].legend(); axes[0].set_ylabel('€/mp')

axes[1].bar(buc['t'], sezon, color=['#2E75B6' if v >= 0 else '#C00000' for v in sezon], alpha=0.8)
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_title('Componenta sezonieră', fontweight='bold')
axes[1].set_ylabel('Variație (€/mp)')

reziduu = buc['pret_mediu_eur_mp'].values - np.nan_to_num(buc_ma, nan=buc['pret_mediu_eur_mp'].mean()) - sezon
axes[2].plot(buc['t'], reziduu, color='gray', linewidth=1.5, marker='o', markersize=4)
axes[2].axhline(0, color='red', linestyle='--', linewidth=1)
axes[2].set_title('Componenta reziduală', fontweight='bold')
axes[2].set_ylabel('Reziduuri')

tick_labels = [f"{r['trimestru']}\n{r['an']}" for _, r in buc.iterrows()]
axes[2].set_xticks(buc['t']); axes[2].set_xticklabels(tick_labels, fontsize=7, rotation=45)

plt.tight_layout()
plt.savefig('outputs/figures/03_descompunere_serie.png', bbox_inches='tight')
plt.show()

# 3. Test stationaritate — Augmented Dickey-Fuller (ADF) manual cu scipy
from scipy import stats as sp_stats

def adf_test_manual(series, name='Serie'):
    y = np.array(series, dtype=float)
    dy = np.diff(y)
    y_lag = y[:-1]
    
    n = len(dy)
    X = np.column_stack([y_lag, np.ones(n)])
    
    beta, res, rank, sv = np.linalg.lstsq(X, dy, rcond=None)
    
    y_hat = X @ beta
    residuals = dy - y_hat
    s2 = np.sum(residuals**2) / (n - 2)
    
    se_beta = np.sqrt(s2 * np.linalg.inv(X.T @ X).diagonal())
    t_stat = beta[0] / se_beta[0]
    
    print(f"\n{'='*45}")
    print(f"ADF Test — {name}")
    print(f"{'='*45}")
    print(f"Statistică t (ADF): {t_stat:.4f}")
    print(f"Valori critice aproximative: -3.75 (1%), -3.00 (5%), -2.63 (10%)")
    if t_stat < -3.00:
        print(">>> Seriestaționară (respingem H0 la 5%)")
    else:
        print(">>> Serie NON-staționară (nu respingem H0)")
    return t_stat

t_buc = adf_test_manual(buc['pret_mediu_eur_mp'], 'București')
t_clj = adf_test_manual(clj['pret_mediu_eur_mp'], 'Cluj-Napoca')

# 4. Diferențiere pentru stationarizare
buc_diff = np.diff(buc['pret_mediu_eur_mp'].values)
clj_diff = np.diff(clj['pret_mediu_eur_mp'].values)

fig, axes = plt.subplots(2, 2, figsize=(13, 7))

axes[0,0].plot(buc['pret_mediu_eur_mp'].values, color='#1F4E79', linewidth=2)
axes[0,0].set_title('București — Serie originală', fontweight='bold')
axes[0,0].set_ylabel('€/mp')

axes[0,1].plot(buc_diff, color='#C00000', linewidth=1.8, marker='o', markersize=4)
axes[0,1].axhline(0, color='gray', linestyle='--')
axes[0,1].set_title('București — Primă diferență (d=1)', fontweight='bold')
axes[0,1].set_ylabel('Δ€/mp')

axes[1,0].plot(clj['pret_mediu_eur_mp'].values, color='#1F4E79', linewidth=2)
axes[1,0].set_title('Cluj-Napoca — Serie originală', fontweight='bold')
axes[1,0].set_ylabel('€/mp')

axes[1,1].plot(clj_diff, color='#C00000', linewidth=1.8, marker='s', markersize=4)
axes[1,1].axhline(0, color='gray', linestyle='--')
axes[1,1].set_title('Cluj-Napoca — Primă diferență (d=1)', fontweight='bold')
axes[1,1].set_ylabel('Δ€/mp')

plt.tight_layout()
plt.savefig('outputs/figures/03_diferentiere.png', bbox_inches='tight')
plt.show()

t_buc_d = adf_test_manual(buc_diff, 'București — prima diferenta')

# 5. ACF și PACF manual
def acf_manual(series, nlags=10):
    n = len(series)
    series = series - np.mean(series)
    var = np.sum(series**2) / n
    acf_vals = []
    for lag in range(0, nlags + 1):
        cov = np.sum(series[:n-lag] * series[lag:]) / n
        acf_vals.append(cov / var)
    return np.array(acf_vals)

def pacf_manual(series, nlags=10):
    from numpy.linalg import lstsq
    n = len(series)
    series = series - np.mean(series)
    pacf_vals = [1.0]
    for lag in range(1, nlags + 1):
        X = np.column_stack([series[lag-k:n-k] for k in range(1, lag+1)])
        y = series[lag:]
        beta, _, _, _ = lstsq(X, y, rcond=None)
        pacf_vals.append(beta[0])
    return np.array(pacf_vals)

buc_d = buc_diff - buc_diff.mean()
acf_vals = acf_manual(buc_d, nlags=10)
pacf_vals = pacf_manual(buc_d, nlags=10)
ci = 1.96 / np.sqrt(len(buc_d))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
lags = np.arange(len(acf_vals))

axes[0].bar(lags, acf_vals, color='#2E75B6', alpha=0.8, width=0.5)
axes[0].axhline(ci, color='red', linestyle='--', linewidth=1.2, label=f'IC 95% (±{ci:.3f})')
axes[0].axhline(-ci, color='red', linestyle='--', linewidth=1.2)
axes[0].axhline(0, color='black', linewidth=0.8)
axes[0].set_title('ACF — București (primă diferență)', fontweight='bold')
axes[0].set_xlabel('Lag (trimestre)'); axes[0].set_ylabel('Autocorelare')
axes[0].legend()

lags_p = np.arange(len(pacf_vals))
axes[1].bar(lags_p, pacf_vals, color='#1F4E79', alpha=0.8, width=0.5)
axes[1].axhline(ci, color='red', linestyle='--', linewidth=1.2, label=f'IC 95% (±{ci:.3f})')
axes[1].axhline(-ci, color='red', linestyle='--', linewidth=1.2)
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_title('PACF — București (primă diferență)', fontweight='bold')
axes[1].set_xlabel('Lag (trimestre)'); axes[1].set_ylabel('Autocorelare parțială')
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/figures/03_acf_pacf.png', bbox_inches='tight')
plt.show()
print(f"Sugestie model: ARIMA(p,d,q) — d=1 (non-stationar)")
print(f"ACF se taie după lag {np.where(np.abs(acf_vals[1:]) < ci)[0][0]+1} → q estimat")
print(f"PACF se taie după lag {np.where(np.abs(pacf_vals[1:]) < ci)[0][0]+1} → p estimat")

# 6. ARIMA(1,1,1) — implementare manuală
class ARIMA111:
    def __init__(self):
        self.params = None
        self.fitted = None
        self.serie = None
        
    def _neg_loglik(self, params, y):
        phi, theta, sigma2 = params[0], params[1], max(params[2], 1e-6)
        n = len(y)
        dy = np.diff(y)
        eps = np.zeros(n - 1)
        ll = 0
        for t in range(1, n - 1):
            ar_term = phi * dy[t-1]
            ma_term = theta * eps[t-1]
            eps[t] = dy[t] - ar_term - ma_term
            ll += -0.5 * np.log(2 * np.pi * sigma2) - eps[t]**2 / (2 * sigma2)
        return -ll
    
    def fit(self, y):
        self.serie = np.array(y, dtype=float)
        x0 = [0.3, 0.3, np.var(np.diff(self.serie))]
        bounds = [(-0.99, 0.99), (-0.99, 0.99), (1e-6, None)]
        result = minimize(self._neg_loglik, x0, args=(self.serie,), 
                         method='L-BFGS-B', bounds=bounds)
        self.params = result.x
        self.phi, self.theta, self.sigma2 = result.x
        
        dy = np.diff(self.serie)
        self.eps = np.zeros(len(dy))
        fitted_diff = np.zeros(len(dy))
        for t in range(1, len(dy)):
            fitted_diff[t] = self.phi * dy[t-1] + self.theta * self.eps[t-1]
            self.eps[t] = dy[t] - fitted_diff[t]
        self.fitted = np.concatenate([[self.serie[0]], self.serie[0] + np.cumsum(fitted_diff)])
        return self
    
    def forecast(self, steps=4):
        y = self.serie
        dy = np.diff(y)
        preds = []
        last_y = y[-1]
        last_dy = dy[-1]
        last_eps = self.eps[-1]
        for h in range(steps):
            dy_pred = self.phi * last_dy + self.theta * last_eps
            y_pred = last_y + dy_pred
            preds.append(y_pred)
            last_y = y_pred
            last_dy = dy_pred
            last_eps = 0.0
        return np.array(preds)
    
    def aic(self):
        n = len(self.serie) - 1
        ll = -self._neg_loglik(self.params, self.serie)
        k = 3
        return -2 * ll + 2 * k

model_buc = ARIMA111()
model_buc.fit(buc['pret_mediu_eur_mp'].values)

print(f"ARIMA(1,1,1) — București")
print(f"  φ (AR1) = {model_buc.phi:.4f}")
print(f"  θ (MA1) = {model_buc.theta:.4f}")
print(f"  σ²      = {model_buc.sigma2:.4f}")
print(f"  AIC     = {model_buc.aic():.4f}")

# 7. Prognoză 4 trimestre (2025)
forecast_buc = model_buc.forecast(steps=4)
forecast_clj = ARIMA111().fit(clj['pret_mediu_eur_mp'].values).forecast(steps=4)
forecast_tim = ARIMA111().fit(tim['pret_mediu_eur_mp'].values).forecast(steps=4)

trimestre_prog = ['T1 2025','T2 2025','T3 2025','T4 2025']

print("Prognoze ARIMA(1,1,1) — preț mediu €/mp:")
print(f"{'Trimestru':<12} {'București':>12} {'Cluj-Napoca':>13} {'Timișoara':>11}")
print("-" * 50)
for i, t in enumerate(trimestre_prog):
    print(f"{t:<12} {forecast_buc[i]:>11.0f}  {forecast_clj[i]:>12.0f}  {forecast_tim[i]:>10.0f}")

sigma_buc = np.sqrt(model_buc.sigma2)
ic_lower = forecast_buc - 1.96 * sigma_buc * np.arange(1, 5)**0.5
ic_upper = forecast_buc + 1.96 * sigma_buc * np.arange(1, 5)**0.5

fig, ax = plt.subplots(figsize=(14, 5))

t_hist = np.arange(len(buc))
ax.plot(t_hist, buc['pret_mediu_eur_mp'].values, 'o-', color='#1F4E79', 
        linewidth=2.5, markersize=5, label='București — date reale')
ax.plot(t_hist, model_buc.fitted, '--', color='#2E75B6', linewidth=1.8, 
        alpha=0.8, label='ARIMA(1,1,1) — fitted')

t_prog = np.arange(len(buc), len(buc) + 4)
ax.plot(t_prog, forecast_buc, 's--', color='#C00000', linewidth=2, 
        markersize=7, label='Prognoză 2025')
ax.fill_between(t_prog, ic_lower, ic_upper, alpha=0.2, color='#C00000', label='IC 95%')

ax.axvline(len(buc) - 0.5, color='gray', linestyle=':', linewidth=1.5)
ax.text(len(buc) - 0.4, buc['pret_mediu_eur_mp'].max(), 'Orizont prognoză', 
        fontsize=9, color='gray', rotation=90, va='top')

all_labels = [f"{r['trimestru']}\n{r['an']}" for _, r in buc.iterrows()] + ['T1\n2025','T2\n2025','T3\n2025','T4\n2025']
ax.set_xticks(np.arange(len(all_labels)))
ax.set_xticklabels(all_labels, fontsize=7, rotation=45)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.set_title('ARIMA(1,1,1) — Prognoză preț mediu €/mp București 2025', 
             fontsize=13, fontweight='bold')
ax.set_ylabel('Preț mediu (€/mp)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('outputs/figures/03_arima_prognoza.png', bbox_inches='tight')
plt.show()

# 8. Trend liniar — regresie pe timp
fig, ax = plt.subplots(figsize=(13, 5))

for serie, label, color, marker in [(buc, 'București', '#1F4E79', 'o'), 
                                     (clj, 'Cluj-Napoca', '#C00000', 's'),
                                     (tim, 'Timișoara', '#2E75B6', '^')]:
    t = serie['t'].values
    y = serie['pret_mediu_eur_mp'].values
    slope, intercept, r, p, se = sp_stats.linregress(t, y)
    trend = slope * t + intercept
    ax.plot(t, y, marker=marker, color=color, linewidth=1.5, markersize=5, label=f'{label}')
    ax.plot(t, trend, linestyle='--', color=color, linewidth=2, alpha=0.6, 
            label=f'{label} trend ({slope:+.0f} €/trim)')

ax.set_title('Trend liniar — prețuri imobiliare pe piețele principale', fontsize=13, fontweight='bold')
ax.set_ylabel('Preț mediu (€/mp)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('outputs/figures/03_trend_liniar.png', bbox_inches='tight')
plt.show()

for serie, label in [(buc, 'București'), (clj, 'Cluj-Napoca'), (tim, 'Timișoara')]:
    slope, intercept, r, p, se = sp_stats.linregress(serie['t'], serie['pret_mediu_eur_mp'])
    print(f"{label}: +{slope:.1f} €/mp/trim | R²={r**2:.3f} | p={p:.4f}")
