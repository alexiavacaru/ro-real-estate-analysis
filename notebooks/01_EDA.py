
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

print("Librăriile sunt importate cu succes.")

from google.colab import files
uploaded = files.upload()
print("Fișiere încărcate:", list(uploaded.keys()))

df   = pd.read_csv('imobiliare_scrape_2024.csv')
hpi  = pd.read_excel('INS_indice_preturi_locuinte.xlsx',
                     sheet_name='HPI_Trimestrial', header=2)
ircc = pd.read_excel('BNR_IRCC_istoric.xlsx',
                     sheet_name='IRCC_Trimestrial', header=2)
aut  = pd.read_csv('autorizatii_constructie_judete.csv')

# Redenumim coloanele cu newline din IRCC și HPI ca să fie mai ușor de lucrat
ircc.columns = ['Trim_Ref', 'Trim_Aplicare', 'IRCC_pct', 'Modificare']
hpi.columns  = ['An', 'Trimestru', 'HPI_Total', 'HPI_Nou',
                 'HPI_Existent', 'Var_Anuala', 'Var_Trimestriala']

print(f"Anunțuri imobiliare: {df.shape[0]} rânduri, {df.shape[1]} coloane")
print(f"HPI: {hpi.shape}")
print(f"IRCC: {ircc.shape}")
print(f"Autorizații: {aut.shape}")

print("=== Primele rânduri din setul imobiliar ===")
display(df.head())

print("\n=== Tipuri de date ===")
print(df.dtypes)

print("\n=== Valori lipsă ===")
print(df.isnull().sum())

print("\n=== Statistici descriptive ===")
display(df.describe().round(1))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Ordonăm orașele după prețul median (descrescător)
ordine_orase = (df.groupby('oras')['pret_eur_mp']
                  .median()
                  .sort_values(ascending=False)
                  .index)

# Boxplot
sns.boxplot(data=df, x='oras', y='pret_eur_mp',
            order=ordine_orase, ax=axes[0], palette='Blues_r')
axes[0].set_title('Distribuție preț €/mp pe orașe', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Oraș')
axes[0].set_ylabel('Preț (€/mp)')
axes[0].tick_params(axis='x', rotation=45)
axes[0].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

# Bar chart cu mediane
mediane = (df.groupby('oras')['pret_eur_mp']
             .median()
             .sort_values(ascending=False))

culori = sns.color_palette('Blues_r', len(mediane))
bare = axes[1].bar(mediane.index, mediane.values, color=culori)
axes[1].set_title('Preț median €/mp pe orașe', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Oraș')
axes[1].set_ylabel('€/mp')
axes[1].tick_params(axis='x', rotation=45)

for bara, val in zip(bare, mediane.values):
    axes[1].text(
        bara.get_x() + bara.get_width() / 2,
        bara.get_height() + 20,
        f'{int(val):,}',
        ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('01_distributie_preturi_orase.png', bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(figsize=(9, 7))

coloane_num = ['pret_eur_mp', 'suprafata_mp', 'camere', 'an_constructie', 'pret_euro']
corr = df[coloane_num].corr()

# Masca triunghiului superior (nu repetăm valorile)
masca = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=masca, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, square=True,
            linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})

ax.set_title('Matricea corelațiilor — variabile numerice',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('01_corelatii.png', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(11, 6))

orase = df['oras'].unique()
paleta = sns.color_palette('tab10', len(orase))

for i, oras in enumerate(orase):
    sub = df[df['oras'] == oras]
    ax.scatter(sub['suprafata_mp'], sub['pret_eur_mp'],
               label=oras, color=paleta[i],
               s=80, alpha=0.75, edgecolors='white', linewidths=0.5)

# Linie de trend
m, b = np.polyfit(df['suprafata_mp'], df['pret_eur_mp'], 1)
x_line = np.linspace(df['suprafata_mp'].min(), df['suprafata_mp'].max(), 100)
ax.plot(x_line, m * x_line + b, 'r--', linewidth=2,
        label=f'Trend global (y={m:.1f}x+{b:.0f})')

ax.set_title('Preț €/mp vs Suprafață utilă', fontsize=13, fontweight='bold')
ax.set_xlabel('Suprafață utilă (mp)')
ax.set_ylabel('Preț (€/mp)')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.savefig('01_pret_vs_suprafata.png', bbox_inches='tight')
plt.show()

# Curățăm datele HPI
hpi_clean = hpi.dropna(subset=['An', 'HPI_Total']).copy()
hpi_clean['An']       = hpi_clean['An'].astype(int).astype(str)
hpi_clean['Trimestru']= hpi_clean['Trimestru'].astype(str).str.strip()
hpi_clean['Perioada'] = hpi_clean['An'] + ' ' + hpi_clean['Trimestru']
hpi_clean['HPI_Total']= pd.to_numeric(hpi_clean['HPI_Total'], errors='coerce')
hpi_plot = hpi_clean.dropna(subset=['HPI_Total']).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(hpi_plot.index, hpi_plot['HPI_Total'],
        color='#1F4E79', linewidth=2.5, marker='o', markersize=4)
ax.fill_between(hpi_plot.index, 100, hpi_plot['HPI_Total'],
                alpha=0.15, color='#1F4E79')
ax.axhline(100, color='gray', linestyle='--', linewidth=1, label='Baza 2015=100')

# Etichete la fiecare 4 trimestre
pas = 4
ax.set_xticks(hpi_plot.index[::pas])
ax.set_xticklabels(hpi_plot['Perioada'].values[::pas],
                   rotation=45, ha='right', fontsize=8)

ax.set_title('Indicele Prețurilor Locuințelor — România 2015–2024\n'
             '(Sursa: INS / Eurostat prc_hpi_q)',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Indice HPI (2015=100)')
ax.legend()
plt.tight_layout()
plt.savefig('01_HPI_evolutie.png', bbox_inches='tight')
plt.show()


ircc_plot = ircc.dropna(subset=['IRCC_pct']).copy()
ircc_plot['IRCC_pct'] = pd.to_numeric(ircc_plot['IRCC_pct'], errors='coerce')
ircc_plot = ircc_plot.dropna(subset=['IRCC_pct']).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(14, 5))

# Barele sunt roșii dacă IRCC >= 5%, altfel albastre
culori_ircc = ['#C00000' if v >= 0.05 else '#2E75B6'
               for v in ircc_plot['IRCC_pct']]

ax.bar(range(len(ircc_plot)),
       ircc_plot['IRCC_pct'] * 100,
       color=culori_ircc, alpha=0.85, width=0.75)

ax.axhline(3, color='orange', linestyle='--', linewidth=1.2, label='Nivel 3%')
ax.set_xticks(range(len(ircc_plot)))
ax.set_xticklabels(ircc_plot['Trim_Ref'], rotation=60, ha='right', fontsize=7.5)
ax.set_title('IRCC Trimestrial — Evoluție Istorică\n'
             '(Sursa: Banca Națională a României)',
             fontsize=13, fontweight='bold')
ax.set_ylabel('IRCC (%)')
ax.legend()
plt.tight_layout()
plt.savefig('01_IRCC_evolutie.png', bbox_inches='tight')
plt.show()


aut24 = (aut[aut['an'] == 2024]
           .sort_values('autorizatii_rezidentiale', ascending=True))

fig, ax = plt.subplots(figsize=(11, 8))

culori_aut = plt.cm.RdYlGn(
    np.linspace(0.2, 0.9, len(aut24)))

bare_h = ax.barh(aut24['judet'],
                 aut24['autorizatii_rezidentiale'],
                 color=culori_aut)

ax.set_xlabel('Nr. autorizații rezidențiale')
ax.set_title('Autorizații de construire rezidențiale pe județe — 2024\n'
             '(Sursa: INS România)',
             fontsize=13, fontweight='bold')

for bara, val in zip(bare_h, aut24['autorizatii_rezidentiale']):
    ax.text(bara.get_width() + 20,
            bara.get_y() + bara.get_height() / 2,
            f'{val:,}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('01_autorizatii_2024.png', bbox_inches='tight')
plt.show()


print("=== Statistici preț €/mp per oraș ===\n")
stats = (df.groupby('oras')['pret_eur_mp']
           .agg(['mean', 'median', 'std', 'min', 'max'])
           .round(0)
           .sort_values('median', ascending=False))
display(stats)

print(f"\nTotal observații în dataset: {len(df):,}")
print(f"Preț mediu național:         {df['pret_eur_mp'].mean():.0f} €/mp")
print(f"Preț median național:        {df['pret_eur_mp'].median():.0f} €/mp")
print(f"Interval preț:               {df['pret_eur_mp'].min():,} — {df['pret_eur_mp'].max():,} €/mp")

print("\nFiguri salvate în directorul curent din Colab.")
