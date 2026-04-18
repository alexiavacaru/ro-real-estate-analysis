

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats as sp_stats
from scipy.optimize import minimize

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 120
sns.set_theme(style='whitegrid')

print("Librăriile sunt importate cu succes.")

df = pd.read_csv('serii_timp_pret_mp.csv')

print("Preview date:")
display(df.head(8))
print(f"\nOrase disponibile: {df['oras'].unique()}")

# Separăm seriile pe cele 3 orașe principale
buc = df[df['oras'] == 'Bucuresti'].copy().reset_index(drop=True)
clj = df[df['oras'] == 'Cluj-Napoca'].copy().reset_index(drop=True)
tim = df[df['oras'] == 'Timisoara'].copy().reset_index(drop=True)

# Adăugăm un index numeric de timp (util pentru regresii)
buc['t'] = range(len(buc))
clj['t'] = range(len(clj))
tim['t'] = range(len(tim))

print(f"\nObservații: București={len(buc)}, Cluj={len(clj)}, Timișoara={len(tim)}")


fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(buc['t'], buc['pret_mediu_eur_mp'],
        'o-', color='#1F4E79', linewidth=2.5, markersize=5, label='București')
ax.plot(clj['t'], clj['pret_mediu_eur_mp'],
        's-', color='#C00000', linewidth=2.5, markersize=5, label='Cluj-Napoca')
ax.plot(tim['t'], tim['pret_mediu_eur_mp'],
        '^-', color='#2E75B6', linewidth=2.5, markersize=5, label='Timișoara')

# Etichete pe axa X: trimestru + an
etichete_x = [f"{r['trimestru']}\n{r['an']}" for _, r in buc.iterrows()]
ax.set_xticks(buc['t'])
ax.set_xticklabels(etichete_x, fontsize=7, rotation=45)
ax.set_title('Evoluția prețului mediu €/mp — Serii trimestriale 2019–2024',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Preț mediu (€/mp)')
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('03_serii_brute.png', bbox_inches='tight')
plt.show()

def medie_mobila(serie, fereastra=4):
    return pd.Series(serie).rolling(window=fereastra, center=True).mean().values

# Calculăm trendul prin medie mobilă de 4 trimestre
buc_ma = medie_mobila(buc['pret_mediu_eur_mp'])

# Componenta sezonieră: diferența față de media generală, grupată pe trimestre
medie_generala = buc['pret_mediu_eur_mp'].mean()
comp_sezon = {}
for t in ['T1', 'T2', 'T3', 'T4']:
    valori_t = buc[buc['trimestru'] == t]['pret_mediu_eur_mp'].values
    comp_sezon[t] = np.mean(valori_t) - medie_generala

sezon = buc['trimestru'].map(comp_sezon).values

# Componenta reziduală = serie - trend - sezon
trend_filled = np.where(np.isnan(buc_ma), medie_generala, buc_ma)
reziduu = buc['pret_mediu_eur_mp'].values - trend_filled - sezon

# Plot descompunere
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(buc['t'], buc['pret_mediu_eur_mp'],
             color='#1F4E79', linewidth=2, label='Serie originală')
axes[0].plot(buc['t'], buc_ma, color='#C00000',
             linewidth=2.5, linestyle='--', label='Trend (MA-4)')
axes[0].set_title('Serie originală + Trend (medie mobilă 4 trimestre)',
                  fontweight='bold')
axes[0].legend()
axes[0].set_ylabel('€/mp')

culori_sezon = ['#2E75B6' if v >= 0 else '#C00000' for v in sezon]
axes[1].bar(buc['t'], sezon, color=culori_sezon, alpha=0.8)
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_title('Componenta sezonieră', fontweight='bold')
axes[1].set_ylabel('Variație (€/mp)')

axes[2].plot(buc['t'], reziduu, color='gray',
             linewidth=1.5, marker='o', markersize=4)
axes[2].axhline(0, color='red', linestyle='--', linewidth=1)
axes[2].set_title('Componenta reziduală', fontweight='bold')
axes[2].set_ylabel('Reziduuri')
axes[2].set_xticks(buc['t'])
axes[2].set_xticklabels(etichete_x, fontsize=7, rotation=45)

plt.tight_layout()
plt.savefig('03_descompunere_serie.png', bbox_inches='tight')
plt.show()


def test_adf_manual(serie, nume='Serie'):
    y      = np.array(serie, dtype=float)
    dy     = np.diff(y)
    y_lag  = y[:-1]
    n      = len(dy)
    X_mat  = np.column_stack([y_lag, np.ones(n)])

    beta, _, _, _ = np.linalg.lstsq(X_mat, dy, rcond=None)
    y_hat         = X_mat @ beta
    reziduale     = dy - y_hat
    s2            = np.sum(reziduale**2) / (n - 2)
    se_beta       = np.sqrt(s2 * np.linalg.inv(X_mat.T @ X_mat).diagonal())
    t_stat        = beta[0] / se_beta[0]

    print(f"\n{'='*45}")
    print(f"ADF Test — {nume}")
    print(f"{'='*45}")
    print(f"Statistică ADF (t): {t_stat:.4f}")
    print(f"Valori critice:     -3.75 (1%) | -3.00 (5%) | -2.63 (10%)")
    if t_stat < -3.00:
        print(">>> Concluzie: Serie STAȚIONARĂ (respingem H0 la 5%)")
    else:
        print(">>> Concluzie: Serie NON-STAȚIONARĂ (nu respingem H0)")
    return t_stat

t_buc = test_adf_manual(buc['pret_mediu_eur_mp'], 'București')
t_clj = test_adf_manual(clj['pret_mediu_eur_mp'], 'Cluj-Napoca')

buc_diff = np.diff(buc['pret_mediu_eur_mp'].values)
clj_diff = np.diff(clj['pret_mediu_eur_mp'].values)

fig, axes = plt.subplots(2, 2, figsize=(13, 7))

axes[0, 0].plot(buc['pret_mediu_eur_mp'].values, color='#1F4E79', linewidth=2)
axes[0, 0].set_title('București — Serie originală', fontweight='bold')
axes[0, 0].set_ylabel('€/mp')

axes[0, 1].plot(buc_diff, color='#C00000', linewidth=1.8,
                marker='o', markersize=4)
axes[0, 1].axhline(0, color='gray', linestyle='--')
axes[0, 1].set_title('București — Primă diferență (d=1)', fontweight='bold')
axes[0, 1].set_ylabel('Δ€/mp')

axes[1, 0].plot(clj['pret_mediu_eur_mp'].values, color='#1F4E79', linewidth=2)
axes[1, 0].set_title('Cluj-Napoca — Serie originală', fontweight='bold')
axes[1, 0].set_ylabel('€/mp')

axes[1, 1].plot(clj_diff, color='#C00000', linewidth=1.8,
                marker='s', markersize=4)
axes[1, 1].axhline(0, color='gray', linestyle='--')
axes[1, 1].set_title('Cluj-Napoca — Primă diferență (d=1)', fontweight='bold')
axes[1, 1].set_ylabel('Δ€/mp')

plt.tight_layout()
plt.savefig('03_diferentiere.png', bbox_inches='tight')
plt.show()

# Verificăm stationaritatea după diferențiere
t_buc_d1 = test_adf_manual(buc_diff, 'București — prima diferență')


def calcul_acf(serie, nr_laguri=10):
    n      = len(serie)
    serie  = serie - np.mean(serie)
    varianta = np.sum(serie**2) / n
    acf_vals = []
    for lag in range(0, nr_laguri + 1):
        cov = np.sum(serie[:n-lag] * serie[lag:]) / n
        acf_vals.append(cov / varianta)
    return np.array(acf_vals)

def calcul_pacf(serie, nr_laguri=10):
    n      = len(serie)
    serie  = serie - np.mean(serie)
    pacf_vals = [1.0]
    for lag in range(1, nr_laguri + 1):
        X_mat = np.column_stack([serie[lag-k:n-k] for k in range(1, lag+1)])
        y_vec = serie[lag:]
        beta, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
        pacf_vals.append(beta[0])
    return np.array(pacf_vals)

buc_d_centrat = buc_diff - buc_diff.mean()
acf_vals  = calcul_acf(buc_d_centrat, nr_laguri=10)
pacf_vals = calcul_pacf(buc_d_centrat, nr_laguri=10)
ci        = 1.96 / np.sqrt(len(buc_d_centrat))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# ACF
laguri = np.arange(len(acf_vals))
axes[0].bar(laguri, acf_vals, color='#2E75B6', alpha=0.8, width=0.5)
axes[0].axhline(ci,  color='red', linestyle='--', linewidth=1.2,
                label=f'IC 95% (±{ci:.3f})')
axes[0].axhline(-ci, color='red', linestyle='--', linewidth=1.2)
axes[0].axhline(0,   color='black', linewidth=0.8)
axes[0].set_title('ACF — București (primă diferență)', fontweight='bold')
axes[0].set_xlabel('Lag (trimestre)')
axes[0].set_ylabel('Autocorelare')
axes[0].legend()

# PACF
laguri_p = np.arange(len(pacf_vals))
axes[1].bar(laguri_p, pacf_vals, color='#1F4E79', alpha=0.8, width=0.5)
axes[1].axhline(ci,  color='red', linestyle='--', linewidth=1.2,
                label=f'IC 95% (±{ci:.3f})')
axes[1].axhline(-ci, color='red', linestyle='--', linewidth=1.2)
axes[1].axhline(0,   color='black', linewidth=0.8)
axes[1].set_title('PACF — București (primă diferență)', fontweight='bold')
axes[1].set_xlabel('Lag (trimestre)')
axes[1].set_ylabel('Autocorelare parțială')
axes[1].legend()

plt.tight_layout()
plt.savefig('03_acf_pacf.png', bbox_inches='tight')
plt.show()

# Recomandare ordin ARIMA
q_est = np.where(np.abs(acf_vals[1:])  < ci)[0]
p_est = np.where(np.abs(pacf_vals[1:]) < ci)[0]
print(f"\nSugestie model: ARIMA(p=1, d=1, q=1)")
print(f"  ACF  se taie după lag {q_est[0]+1 if len(q_est) > 0 else '?'} → q estimat")
print(f"  PACF se taie după lag {p_est[0]+1 if len(p_est) > 0 else '?'} → p estimat")

class ARIMA111:
    """
    Implementare simplificată a modelului ARIMA(1,1,1).
    Estimare prin maximizarea log-verosimilității (MLE).
    """

    def __init__(self):
        self.params  = None
        self.fitted  = None
        self.serie   = None
        self.eps     = None

    def _neg_loglik(self, params, y):
        phi, theta, sigma2 = params[0], params[1], max(params[2], 1e-6)
        dy  = np.diff(y)
        n   = len(dy)
        eps = np.zeros(n)
        ll  = 0.0
        for t in range(1, n):
            eps[t] = dy[t] - phi * dy[t-1] - theta * eps[t-1]
            ll    += -0.5 * np.log(2 * np.pi * sigma2) - eps[t]**2 / (2 * sigma2)
        return -ll

    def fit(self, y):
        self.serie = np.array(y, dtype=float)
        dy         = np.diff(self.serie)
        x0         = [0.3, 0.3, np.var(dy)]
        bounds     = [(-0.99, 0.99), (-0.99, 0.99), (1e-6, None)]
        rez        = minimize(self._neg_loglik, x0, args=(self.serie,),
                              method='L-BFGS-B', bounds=bounds)
        self.params = rez.x
        self.phi, self.theta, self.sigma2 = rez.x

        # Calculăm valorile fitted
        eps_fit   = np.zeros(len(dy))
        fitted_dy = np.zeros(len(dy))
        for t in range(1, len(dy)):
            fitted_dy[t] = self.phi * dy[t-1] + self.theta * eps_fit[t-1]
            eps_fit[t]   = dy[t] - fitted_dy[t]

        self.eps    = eps_fit
        self.fitted = np.concatenate([
            [self.serie[0]],
            self.serie[0] + np.cumsum(fitted_dy)
        ])
        return self

    def forecast(self, steps=4):
        dy      = np.diff(self.serie)
        preds   = []
        last_y  = self.serie[-1]
        last_dy = dy[-1]
        last_eps = self.eps[-1]

        for h in range(steps):
            dy_pred  = self.phi * last_dy + self.theta * last_eps
            y_pred   = last_y + dy_pred
            preds.append(y_pred)
            last_y   = y_pred
            last_dy  = dy_pred
            last_eps = 0.0  # eroarea viitoare e necunoscută → 0

        return np.array(preds)

    def aic(self):
        ll = -self._neg_loglik(self.params, self.serie)
        return -2 * ll + 2 * 3   # k=3 parametri: phi, theta, sigma2

print("Clasa ARIMA(1,1,1) definită cu succes.")

model_buc = ARIMA111().fit(buc['pret_mediu_eur_mp'].values)
model_clj = ARIMA111().fit(clj['pret_mediu_eur_mp'].values)
model_tim = ARIMA111().fit(tim['pret_mediu_eur_mp'].values)

print("=== Parametri ARIMA(1,1,1) ===\n")
for model, oras in [(model_buc, 'București'),
                    (model_clj, 'Cluj-Napoca'),
                    (model_tim, 'Timișoara')]:
    print(f"{oras}:")
    print(f"  φ (AR1) = {model.phi:.4f}")
    print(f"  θ (MA1) = {model.theta:.4f}")
    print(f"  σ²      = {model.sigma2:.4f}")
    print(f"  AIC     = {model.aic():.4f}\n")

forecast_buc = model_buc.forecast(steps=4)
forecast_clj = model_clj.forecast(steps=4)
forecast_tim = model_tim.forecast(steps=4)

trim_prog = ['T1 2025', 'T2 2025', 'T3 2025', 'T4 2025']

print("=== Prognoze ARIMA(1,1,1) — Preț mediu €/mp ===\n")
print(f"{'Trimestru':<12} {'București':>12} {'Cluj-Napoca':>13} {'Timișoara':>11}")
print("-" * 52)
for i, t in enumerate(trim_prog):
    print(f"{t:<12} {forecast_buc[i]:>11.0f}  {forecast_clj[i]:>12.0f}  {forecast_tim[i]:>10.0f}")


sigma_buc = np.sqrt(model_buc.sigma2)
orizont   = np.arange(1, 5)
ic_lower  = forecast_buc - 1.96 * sigma_buc * orizont**0.5
ic_upper  = forecast_buc + 1.96 * sigma_buc * orizont**0.5

fig, ax = plt.subplots(figsize=(14, 5))

t_hist = np.arange(len(buc))
ax.plot(t_hist, buc['pret_mediu_eur_mp'].values,
        'o-', color='#1F4E79', linewidth=2.5, markersize=5,
        label='București — date reale')
ax.plot(t_hist, model_buc.fitted,
        '--', color='#2E75B6', linewidth=1.8, alpha=0.8,
        label='ARIMA(1,1,1) — fitted')

t_prog = np.arange(len(buc), len(buc) + 4)
ax.plot(t_prog, forecast_buc,
        's--', color='#C00000', linewidth=2, markersize=7,
        label='Prognoză 2025')
ax.fill_between(t_prog, ic_lower, ic_upper,
                alpha=0.2, color='#C00000', label='Interval de încredere 95%')

# Linie verticală care separă datele reale de prognoze
ax.axvline(len(buc) - 0.5, color='gray', linestyle=':', linewidth=1.5)
ax.text(len(buc) - 0.3, buc['pret_mediu_eur_mp'].max() * 0.97,
        'Orizont\nprognoză', fontsize=8.5, color='gray',
        rotation=90, va='top')

# Etichete axa X
etich_complet = (
    [f"{r['trimestru']}\n{r['an']}" for _, r in buc.iterrows()] +
    ['T1\n2025', 'T2\n2025', 'T3\n2025', 'T4\n2025']
)
ax.set_xticks(np.arange(len(etich_complet)))
ax.set_xticklabels(etich_complet, fontsize=7, rotation=45)
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.set_title('ARIMA(1,1,1) — Prognoză preț mediu €/mp București 2025',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Preț mediu (€/mp)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('03_arima_prognoza.png', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(13, 5))

serii_info = [
    (buc, 'București',    '#1F4E79', 'o'),
    (clj, 'Cluj-Napoca',  '#C00000', 's'),
    (tim, 'Timișoara',    '#2E75B6', '^'),
]

for serie, label, culoare, marker in serii_info:
    t = serie['t'].values
    y = serie['pret_mediu_eur_mp'].values
    slope, intercept, r, p, se = sp_stats.linregress(t, y)
    trend = slope * t + intercept

    ax.plot(t, y, marker=marker, color=culoare,
            linewidth=1.5, markersize=5, label=label)
    ax.plot(t, trend, linestyle='--', color=culoare, linewidth=2, alpha=0.6,
            label=f'{label} trend ({slope:+.0f} €/trim)')

ax.set_title('Trend liniar — prețuri imobiliare pe piețele principale',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Preț mediu (€/mp)')
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('03_trend_liniar.png', bbox_inches='tight')
plt.show()

# Afișăm parametrii trendului
print("\n=== Parametrii Trendului Liniar ===")
for serie, label in [(buc, 'București'), (clj, 'Cluj-Napoca'), (tim, 'Timișoara')]:
    slope, intercept, r, p, se = sp_stats.linregress(
        serie['t'], serie['pret_mediu_eur_mp'])
    print(f"{label}: +{slope:.1f} €/mp pe trimestru | R²={r**2:.3f} | p={p:.4f}")
