
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.cluster         import KMeans
from sklearn.preprocessing   import StandardScaler
from sklearn.decomposition   import PCA
from sklearn.metrics         import (silhouette_score,
                                     davies_bouldin_score,
                                     calinski_harabasz_score)

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 120
np.random.seed(42)
sns.set_theme(style='whitegrid')

print("Librăriile sunt importate cu succes.")

cluster_df = pd.DataFrame({
    'oras': [
        'Bucuresti', 'Cluj-Napoca', 'Timisoara',
        'Iasi', 'Brasov', 'Constanta', 'Craiova', 'Oradea'
    ],
    'pret_mediu_eur_mp':   [1748, 2658, 1929, 1636, 2102, 1606, 1563, 1760],
    'pret_std':            [280,   95,  150,   90,   70,   90,   15,   22],
    'autorizatii_2024':    [4548, 3246, 2957, 2828, 1648, 1677, 1387, 2396],
    'var_autorizatii_pct': [4.7,   0.8, 10.2,  2.8, -5.1,  1.0, -1.5,  0.5],
    'suprafata_medie_mp':  [64,    60,   57,   57,   68,   65,   67,   68],
    'nr_tranzactii_est':   [18500, 6800, 5200, 4800, 3200, 3500, 2800, 3100]
})

print("Date pentru clustering:")
display(cluster_df)

features = [
    'pret_mediu_eur_mp',
    'pret_std',
    'autorizatii_2024',
    'var_autorizatii_pct',
    'suprafata_medie_mp',
    'nr_tranzactii_est'
]

X = cluster_df[features].values

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

print("Date standardizate (primele 3 rânduri):")
print(pd.DataFrame(X_sc, columns=features).head(3).round(3))

inertii    = []
sil_scores = []
K_range    = range(2, 8)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
    km.fit(X_sc)
    inertii.append(km.inertia_)

    labels = km.predict(X_sc)
    sil_scores.append(silhouette_score(X_sc, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Grafic 1: Metoda Cotului (Elbow)
axes[0].plot(K_range, inertii, 'bo-', linewidth=2.5, markersize=8)
axes[0].set_xlabel('Număr de clustere K')
axes[0].set_ylabel('Inerție (WCSS)')
axes[0].set_title('Metoda Cotului — alegerea K optim', fontweight='bold')
axes[0].set_xticks(list(K_range))
for k, iner in zip(K_range, inertii):
    axes[0].annotate(f'{iner:.1f}', (k, iner),
                     textcoords='offset points', xytext=(0, 10),
                     ha='center', fontsize=8)

# Grafic 2: Scor Silhouette
axes[1].plot(K_range, sil_scores, 'o-', linewidth=2.5, markersize=8,
             color='#C00000')
axes[1].set_xlabel('Număr de clustere K')
axes[1].set_ylabel('Coeficient Silhouette')
axes[1].set_title('Scor Silhouette — calitatea clusterizării', fontweight='bold')
axes[1].set_xticks(list(K_range))

k_optim = list(K_range)[np.argmax(sil_scores)]
axes[1].axvline(k_optim, color='gray', linestyle='--', linewidth=1.5,
                label=f'K optim = {k_optim}')
axes[1].legend()

plt.tight_layout()
plt.savefig('04_elbow_silhouette.png', bbox_inches='tight')
plt.show()

print(f"\nK optim după Silhouette: {k_optim}")
print(f"Scoruri Silhouette: { {k: round(s, 4) for k, s in zip(K_range, sil_scores)} }")


K = 3
kmeans = KMeans(n_clusters=K, init='k-means++',
                n_init=50, max_iter=500, random_state=42)
cluster_df['cluster'] = kmeans.fit_predict(X_sc)

sil = silhouette_score(X_sc, cluster_df['cluster'])
db  = davies_bouldin_score(X_sc, cluster_df['cluster'])
ch  = calinski_harabasz_score(X_sc, cluster_df['cluster'])

print(f"=== Metrici calitate K-Means (K={K}) ===")
print(f"Silhouette Score     : {sil:.4f}  (↑ mai aproape de 1 = mai bun)")
print(f"Davies-Bouldin Index : {db:.4f}  (↓ mai mic = mai bun)")
print(f"Calinski-Harabasz    : {ch:.4f}  (↑ mai mare = mai bun)")

print("\nAtribuire orașe → clustere:")
display(cluster_df[['oras', 'cluster']])


pca  = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_sc)

cluster_df['PC1'] = X_pca[:, 0]
cluster_df['PC2'] = X_pca[:, 1]

var1 = pca.explained_variance_ratio_[0] * 100
var2 = pca.explained_variance_ratio_[1] * 100

fig, ax = plt.subplots(figsize=(10, 7))

paleta    = {0: '#1F4E79', 1: '#C00000', 2: '#70AD47'}
num_clust = sorted(cluster_df['cluster'].unique())

for cl in num_clust:
    sub = cluster_df[cluster_df['cluster'] == cl]
    ax.scatter(sub['PC1'], sub['PC2'],
               color=paleta[cl], s=200,
               label=f'Cluster {cl}',
               edgecolors='white', linewidths=1.5,
               zorder=5, alpha=0.9)

    # Etichete cu numele orașului
    for _, rand in sub.iterrows():
        ax.annotate(
            rand['oras'], (rand['PC1'], rand['PC2']),
            textcoords='offset points', xytext=(8, 5),
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      alpha=0.7, edgecolor='gray'))

# Centroizi
centre_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centre_pca[:, 0], centre_pca[:, 1],
           marker='X', s=350,
           color=[paleta[i] for i in range(K)],
           edgecolors='black', linewidths=1.5,
           zorder=10, label='Centroizi')

ax.set_xlabel(f'PC1 ({var1:.1f}% varianță explicată)', fontsize=11)
ax.set_ylabel(f'PC2 ({var2:.1f}% varianță explicată)', fontsize=11)
ax.set_title(
    f'K-Means (K={K}) — Vizualizare PCA 2D\n'
    f'Total varianță explicată: {var1+var2:.1f}%',
    fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('04_kmeans_pca.png', bbox_inches='tight')
plt.show()

profil = cluster_df.groupby('cluster')[features].mean()

print("=== Profil mediu pe clustere ===")
display(profil.round(1))

etichete_features = [
    'Preț mediu (€/mp)',
    'Std preț (€/mp)',
    'Autorizații 2024',
    'Variație autorizații (%)',
    'Suprafață medie (mp)',
    'Nr. tranzacții est.'
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
culori_cl = ['#1F4E79', '#C00000', '#70AD47']

for i, (feat, label) in enumerate(zip(features, etichete_features)):
    valori = [cluster_df[cluster_df['cluster'] == cl][feat].mean()
              for cl in range(K)]
    bare = axes[i].bar(range(K), valori,
                       color=culori_cl, alpha=0.85,
                       width=0.5, edgecolor='white')
    axes[i].set_xticks(range(K))
    axes[i].set_xticklabels([f'Cluster {j}' for j in range(K)], fontsize=9)
    axes[i].set_title(label, fontweight='bold', fontsize=10)

    for bara, v in zip(bare, valori):
        axes[i].text(bara.get_x() + bara.get_width() / 2,
                     bara.get_height() * 1.02,
                     f'{v:,.0f}',
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.suptitle('Profilul clusterelor — valori medii per feature',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('04_profil_clustere.png', bbox_inches='tight')
plt.show()

profil_sc = pd.DataFrame(
    scaler.transform(profil),
    index=[f'Cluster {i}' for i in range(K)],
    columns=etichete_features
)

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(profil_sc, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, linewidths=0.5,
            ax=ax, cbar_kws={'label': 'Z-score'})
ax.set_title('Heatmap profil clustere (valori standardizate)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('04_heatmap_clustere.png', bbox_inches='tight')
plt.show()


etichete_clustere = {
    0: 'Piață maturizată / premium',
    1: 'Piață dinamică / capitală',
    2: 'Piață emergentă'
}

print("=" * 58)
print("INTERPRETARE CLUSTERE K-Means (K=3)")
print("=" * 58)

for cl in range(K):
    orase_cl  = cluster_df[cluster_df['cluster'] == cl]['oras'].tolist()
    pret_med  = cluster_df[cluster_df['cluster'] == cl]['pret_mediu_eur_mp'].mean()
    aut_med   = cluster_df[cluster_df['cluster'] == cl]['autorizatii_2024'].mean()
    tranz_med = cluster_df[cluster_df['cluster'] == cl]['nr_tranzactii_est'].mean()

    print(f"\nCluster {cl} — {etichete_clustere[cl]}")
    print(f"  Orașe:             {', '.join(orase_cl)}")
    print(f"  Preț mediu:        {pret_med:,.0f} €/mp")
    print(f"  Autorizații 2024:  {aut_med:,.0f}")
    print(f"  Tranzacții estim.: {tranz_med:,.0f}")

print(f"\nSilhouette Score final: {sil:.4f}")
print("(Valori > 0.5 indică o clusterizare solidă)")

