import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_theme(style='whitegrid', palette='muted')

df = pd.read_csv('data/raw/imobiliare_scrape_2024.csv')
hpi = pd.read_excel('data/raw/INS_indice_preturi_locuinte.xlsx', sheet_name='HPI_Trimestrial', header=2)
ircc = pd.read_excel('data/raw/BNR_IRCC_istoric.xlsx', sheet_name='IRCC_Trimestrial', header=2)
aut = pd.read_csv('data/raw/autorizatii_constructie_judete.csv')

print(df.shape, hpi.shape, ircc.shape, aut.shape)

df.info()
df.describe()
df.isnull().sum()

# 1. Distribuția prețurilor €/mp pe orașe
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

order = df.groupby('oras')['pret_eur_mp'].median().sort_values(ascending=False).index

sns.boxplot(data=df, x='oras', y='pret_eur_mp', order=order, ax=axes[0], palette='Blues_r')
axes[0].set_title('Distribuție preț €/mp pe orașe', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Oraș')
axes[0].set_ylabel('Preț (€/mp)')
axes[0].tick_params(axis='x', rotation=45)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

medians = df.groupby('oras')['pret_eur_mp'].median().sort_values(ascending=False)
bars = axes[1].bar(medians.index, medians.values, color=sns.color_palette('Blues_r', len(medians)))
axes[1].set_title('Preț median €/mp pe orașe', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Oraș')
axes[1].set_ylabel('€/mp')
axes[1].tick_params(axis='x', rotation=45)
for bar, val in zip(bars, medians.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{int(val):,}', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/figures/01_distributie_preturi_orase.png', bbox_inches='tight')
plt.show()

# 2. Corelație variabile numerice
fig, ax = plt.subplots(figsize=(9, 7))
num_cols = ['pret_eur_mp', 'suprafata_mp', 'camere', 'an_constructie', 'pret_euro']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Matricea corelațiilor — variabile numerice', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/01_corelatii.png', bbox_inches='tight')
plt.show()

# 3. Preț €/mp vs Suprafață
fig, ax = plt.subplots(figsize=(10, 6))
oras_list = df['oras'].unique()
palette = sns.color_palette('tab10', len(oras_list))
for i, oras in enumerate(oras_list):
    sub = df[df['oras'] == oras]
    ax.scatter(sub['suprafata_mp'], sub['pret_eur_mp'], 
               label=oras, color=palette[i], s=80, alpha=0.75, edgecolors='white', linewidths=0.5)

m, b = np.polyfit(df['suprafata_mp'], df['pret_eur_mp'], 1)
x_line = np.linspace(df['suprafata_mp'].min(), df['suprafata_mp'].max(), 100)
ax.plot(x_line, m * x_line + b, 'r--', linewidth=2, label=f'Trend (y={m:.1f}x+{b:.0f})')

ax.set_title('Preț €/mp vs Suprafață utilă', fontsize=13, fontweight='bold')
ax.set_xlabel('Suprafață utilă (mp)')
ax.set_ylabel('Preț (€/mp)')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
plt.savefig('outputs/figures/01_pret_vs_suprafata.png', bbox_inches='tight')
plt.show()

# 4. HPI România 2015–2024 (baza 2015=100)
hpi_clean = hpi[hpi.iloc[:,0].notna()].copy()
hpi_clean.columns = hpi_clean.columns.str.strip()
hpi_clean = hpi_clean[hpi_clean.iloc[:,0].astype(str).str.match(r'^\d{4}$')].copy()
hpi_clean.columns.values[0] = 'An'
hpi_clean.columns.values[1] = 'Trim'
hpi_clean.columns.values[2] = 'HPI_Total'
hpi_clean['An'] = hpi_clean['An'].astype(str)
hpi_clean['Trim'] = hpi_clean['Trim'].astype(str)
hpi_clean['Perioada'] = hpi_clean['An'] + '-' + hpi_clean['Trim']
hpi_clean['HPI_Total'] = pd.to_numeric(hpi_clean['HPI_Total'], errors='coerce')
hpi_plot = hpi_clean.dropna(subset=['HPI_Total'])

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(range(len(hpi_plot)), hpi_plot['HPI_Total'].values, 
        color='#1F4E79', linewidth=2.5, marker='o', markersize=4)
ax.fill_between(range(len(hpi_plot)), 100, hpi_plot['HPI_Total'].values, alpha=0.15, color='#1F4E79')
ax.axhline(100, color='gray', linestyle='--', linewidth=1, label='Baza 2015=100')

step = 4
ax.set_xticks(range(0, len(hpi_plot), step))
ax.set_xticklabels(hpi_plot['Perioada'].values[::step], rotation=45, ha='right', fontsize=8)
ax.set_title('Indicele Prețurilor Locuințelor — România 2015–2024\n(Sursa: INS → Eurostat prc_hpi_q)', fontsize=13, fontweight='bold')
ax.set_ylabel('Indice HPI (2015=100)')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/figures/01_HPI_evolutie.png', bbox_inches='tight')
plt.show()

# 5. IRCC trimestrial 2019–2025
ircc_clean = ircc.copy()
ircc_clean.columns = ircc_clean.columns.str.strip()
ircc_clean = ircc_clean[ircc_clean.iloc[:,0].notna()].copy()
ircc_clean.columns.values[0] = 'Trim_Ref'
ircc_clean.columns.values[2] = 'IRCC_pct'
ircc_clean['IRCC_pct'] = pd.to_numeric(ircc_clean['IRCC_pct'], errors='coerce')
ircc_plot = ircc_clean.dropna(subset=['IRCC_pct'])

fig, ax = plt.subplots(figsize=(14, 5))
colors = ['#C00000' if v >= 0.05 else '#2E75B6' for v in ircc_plot['IRCC_pct'].values]
bars = ax.bar(range(len(ircc_plot)), ircc_plot['IRCC_pct'].values * 100, color=colors, alpha=0.85, width=0.75)
ax.axhline(3, color='orange', linestyle='--', linewidth=1.2, label='Nivel 3%')
ax.set_xticks(range(len(ircc_plot)))
ax.set_xticklabels(ircc_plot['Trim_Ref'].values, rotation=60, ha='right', fontsize=7.5)
ax.set_title('IRCC Trimestrial — Evoluție Istorică\n(Sursa: Banca Națională a României)', fontsize=13, fontweight='bold')
ax.set_ylabel('IRCC (%)')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/figures/01_IRCC_evolutie.png', bbox_inches='tight')
plt.show()

# 6. Autorizații construire rezidențiale pe județe (2024)
aut24 = aut[aut['an'] == 2024].sort_values('autorizatii_rezidentiale', ascending=True)

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(aut24['judet'], aut24['autorizatii_rezidentiale'],
               color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(aut24))))
ax.set_xlabel('Nr. autorizații rezidențiale')
ax.set_title('Autorizații de construire rezidențiale pe județe — 2024\n(Sursa: INS România)', fontsize=13, fontweight='bold')
for bar, val in zip(bars, aut24['autorizatii_rezidentiale']):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2, f'{val:,}', 
            va='center', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/figures/01_autorizatii_2024.png', bbox_inches='tight')
plt.show()

# 7. Statistici descriptive finale
print("=== Statistici preț €/mp per oraș ===")
print(df.groupby('oras')['pret_eur_mp'].agg(['mean','median','std','min','max']).round(0).to_string())
print()
print(f"Total observații: {len(df)}")
print(f"Preț mediu national: {df['pret_eur_mp'].mean():.0f} €/mp")
print(f"Preț median național: {df['pret_eur_mp'].median():.0f} €/mp")
