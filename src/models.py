# aduna aici functiile de modelare din notebook-urile 02, 03 si 04, ca sa nu
# mai copiez acelasi cod si in dashboard.py. Practic e acelasi cod din
# notebook-uri, doar bagat in functii.

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. REGRESIE MULTIPLA (din 02_regresie_multipla.py)

FEATURES = ['suprafata_mp', 'camere', 'vechime', 'oras_enc', 'regiune_enc',
            'HPI_trim', 'IRCC_trim_pct', 'autorizatii_judet_an']


def regresie_preturi(df, random_state=42):
    """antreneaza OLS, Ridge si Lasso pe dataset_final si intoarce un dict
    cu rezultatele, ca sa le pot afisa oriunde am nevoie (notebook sau dashboard)."""

    df = df.copy()
    df['oras_enc'] = LabelEncoder().fit_transform(df['oras'])
    df['regiune_enc'] = LabelEncoder().fit_transform(df['regiune'])
    df['vechime'] = 2024 - df['an_constructie']

    X = df[FEATURES]
    y = df['pret_eur_mp']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    alphas = [0.01, 0.1, 1, 10, 100, 500]

    ols = LinearRegression()
    ols.fit(X_train_sc, y_train)
    y_pred_ols = ols.predict(X_test_sc)

    ridge = RidgeCV(alphas=alphas, cv=5, scoring='r2')
    ridge.fit(X_train_sc, y_train)
    y_pred_ridge = ridge.predict(X_test_sc)

    lasso = LassoCV(alphas=alphas, cv=5, max_iter=5000, random_state=random_state)
    lasso.fit(X_train_sc, y_train)
    y_pred_lasso = lasso.predict(X_test_sc)

    rezultate = pd.DataFrame({
        'Model': ['OLS', 'Ridge', 'Lasso'],
        'R2': [r2_score(y_test, y_pred_ols), r2_score(y_test, y_pred_ridge), r2_score(y_test, y_pred_lasso)],
        'RMSE': [np.sqrt(mean_squared_error(y_test, y_pred_ols)),
                 np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
                 np.sqrt(mean_squared_error(y_test, y_pred_lasso))],
        'MAE': [mean_absolute_error(y_test, y_pred_ols),
                mean_absolute_error(y_test, y_pred_ridge),
                mean_absolute_error(y_test, y_pred_lasso)],
    })

    coef_ols = pd.DataFrame({'Feature': FEATURES, 'Coeficient': ols.coef_})
    coef_ols = coef_ols.sort_values('Coeficient', key=abs, ascending=False)

    return {
        'rezultate': rezultate,
        'coef_ols': coef_ols,
        'ols': ols, 'ridge': ridge, 'lasso': lasso,
        'y_test': y_test, 'y_pred_ols': y_pred_ols,
    }


# 2. SERIE DE TIMP - ARIMA(1,1,1) manual (din 03_serii_de_timp.py)

class ARIMA111:
    # implementare simpla ARIMA(1,1,1), fara statsmodels - facuta pe cont
    # propriu pentru proiect, prin minimizarea log-verosimilitatii negative.

    def __init__(self):
        self.params = None
        self.fitted = None
        self.serie = None
        self.eps = None

    def _neg_loglik(self, params, y):
        phi, theta, sigma2 = params[0], params[1], max(params[2], 1e-6)
        dy = np.diff(y)
        eps = np.zeros(len(dy) - 1)
        ll = 0
        for t in range(1, len(dy) - 1):
            ar_term = phi * dy[t - 1]
            ma_term = theta * eps[t - 1]
            eps[t] = dy[t] - ar_term - ma_term
            ll += -0.5 * np.log(2 * np.pi * sigma2) - eps[t] ** 2 / (2 * sigma2)
        return -ll

    def fit(self, y):
        self.serie = np.array(y, dtype=float)
        x0 = [0.3, 0.3, np.var(np.diff(self.serie))]
        bounds = [(-0.99, 0.99), (-0.99, 0.99), (1e-6, None)]
        result = minimize(self._neg_loglik, x0, args=(self.serie,), method='L-BFGS-B', bounds=bounds)
        self.params = result.x
        self.phi, self.theta, self.sigma2 = result.x

        dy = np.diff(self.serie)
        self.eps = np.zeros(len(dy))
        fitted_diff = np.zeros(len(dy))
        for t in range(1, len(dy)):
            fitted_diff[t] = self.phi * dy[t - 1] + self.theta * self.eps[t - 1]
            self.eps[t] = dy[t] - fitted_diff[t]
        self.fitted = np.concatenate([[self.serie[0]], self.serie[0] + np.cumsum(fitted_diff)])
        return self

    def forecast(self, steps=4):
        dy = np.diff(self.serie)
        preds = []
        last_y = self.serie[-1]
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
        ll = -self._neg_loglik(self.params, self.serie)
        k = 3
        return -2 * ll + 2 * k


def prognoza_oras(serii, oras, pasi=4):
    # serii = dataframe-ul din serii_timp_pret_mp.csv
    sub = serii[serii['oras'] == oras].reset_index(drop=True)
    model = ARIMA111()
    model.fit(sub['pret_mediu_eur_mp'].values)
    forecast = model.forecast(steps=pasi)
    return sub, model, forecast


# 3. CLUSTERING K-MEANS (din 04_clustering_kmeans.py)

FEATURES_CLUSTER = ['pret_mediu_eur_mp', 'pret_std', 'autorizatii_2024',
                     'var_autorizatii_pct', 'suprafata_medie_mp', 'nr_tranzactii_est']

# aceleasi date agregate manual pe care le foloseam si in notebook-ul 04
CLUSTER_DF = pd.DataFrame({
    'oras': ['Bucuresti', 'Cluj-Napoca', 'Timisoara', 'Iasi', 'Brasov', 'Constanta', 'Craiova', 'Oradea'],
    'pret_mediu_eur_mp': [1748, 2658, 1929, 1636, 2102, 1606, 1563, 1760],
    'pret_std': [280, 95, 150, 90, 70, 90, 15, 22],
    'autorizatii_2024': [4548, 3246, 2957, 2828, 1648, 1677, 1387, 2396],
    'var_autorizatii_pct': [4.7, 0.8, 10.2, 2.8, -5.1, 1.0, -1.5, 0.5],
    'suprafata_medie_mp': [64, 60, 57, 57, 68, 65, 67, 68],
    'nr_tranzactii_est': [18500, 6800, 5200, 4800, 3200, 3500, 2800, 3100],
})


def clustere_kmeans(cluster_df=None, k=3, random_state=42):
    if cluster_df is None:
        cluster_df = CLUSTER_DF.copy()
    else:
        cluster_df = cluster_df.copy()

    X = cluster_df[FEATURES_CLUSTER].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=50, max_iter=500, random_state=random_state)
    cluster_df['cluster'] = kmeans.fit_predict(X_sc)

    sil = silhouette_score(X_sc, cluster_df['cluster'])
    db = davies_bouldin_score(X_sc, cluster_df['cluster'])
    ch = calinski_harabasz_score(X_sc, cluster_df['cluster'])

    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_sc)
    cluster_df['PC1'] = X_pca[:, 0]
    cluster_df['PC2'] = X_pca[:, 1]

    metrici = {'silhouette': sil, 'davies_bouldin': db, 'calinski_harabasz': ch}
    return cluster_df, kmeans, metrici


def elbow_si_silhouette(cluster_df=None, k_range=range(2, 8), random_state=42):
    if cluster_df is None:
        cluster_df = CLUSTER_DF.copy()
    X = cluster_df[FEATURES_CLUSTER].values
    X_sc = StandardScaler().fit_transform(X)

    inertii = []
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=random_state)
        labels = km.fit_predict(X_sc)
        inertii.append(km.inertia_)
        sil_scores.append(silhouette_score(X_sc, labels))

    return pd.DataFrame({'k': list(k_range), 'inertie': inertii, 'silhouette': sil_scores})
