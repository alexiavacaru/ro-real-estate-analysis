# curata datele brute din data/raw si le combina intr-un singur fisier,
# dataset_final.csv, folosit apoi in notebook-urile de regresie/clustering.

import pandas as pd
import numpy as np

# ---- 1. incarcare date brute ----------------------------------------------

df = pd.read_csv('data/raw/imobiliare_scrape_2024.csv')
hpi = pd.read_excel('data/raw/INS_indice_preturi_locuinte.xlsx', sheet_name='HPI_Trimestrial', header=2)
ircc = pd.read_excel('data/raw/BNR_IRCC_istoric.xlsx', sheet_name='IRCC_Trimestrial', header=2)
aut = pd.read_csv('data/raw/autorizatii_constructie_judete.csv')

print(df.shape, hpi.shape, ircc.shape, aut.shape)

# ---- 2. curatare anunturi (aceleasi filtre ca in 01_EDA) -------------------

df = df.dropna(subset=['pret_eur_mp', 'suprafata_mp', 'oras'])
df = df[df['suprafata_mp'].between(15, 400)]
df = df[df['pret_eur_mp'].between(300, 8000)]
df = df[df['camere'].between(1, 8)]
df = df[df['an_constructie'].between(1900, 2024)]
df = df.reset_index(drop=True)

print('dupa curatare:', df.shape)

# ---- 3. adaugare judet si regiune pe baza orasului --------------------------
# (nu era in fisierul brut, l-am adaugat manual dupa lista oraselor din studiu)

judet_dupa_oras = {
    'Bucuresti': 'Ilfov',
    'Cluj-Napoca': 'Cluj',
    'Timisoara': 'Timis',
    'Iasi': 'Iasi',
    'Brasov': 'Brasov',
    'Constanta': 'Constanta',
    'Craiova': 'Dolj',
    'Oradea': 'Bihor',
}

regiune_dupa_oras = {
    'Bucuresti': 'Bucuresti-Ilfov',
    'Cluj-Napoca': 'Nord-Vest',
    'Timisoara': 'Vest',
    'Iasi': 'Nord-Est',
    'Brasov': 'Centru',
    'Constanta': 'Sud-Est',
    'Craiova': 'Sud-Vest Oltenia',
    'Oradea': 'Nord-Vest',
}

df['judet'] = df['oras'].map(judet_dupa_oras)
df['regiune'] = df['oras'].map(regiune_dupa_oras)
df = df.rename(columns={'tip_proprietate': 'tip_prop'})

# ---- 4. din "luna_anunt" (ex: "Ian 2024") facem data + trimestrul ----------

luna_cod = {'Ian': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'Mai': '05', 'Iun': '06',
            'Iul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Noi': '11', 'Dec': '12'}
luna_trim = {'Ian': 'T1', 'Feb': 'T1', 'Mar': 'T1', 'Apr': 'T2', 'Mai': 'T2', 'Iun': 'T2',
             'Iul': 'T3', 'Aug': 'T3', 'Sep': 'T3', 'Oct': 'T4', 'Noi': 'T4', 'Dec': 'T4'}

luna_txt = df['luna_anunt'].str.split(' ').str[0]
an_txt = df['luna_anunt'].str.split(' ').str[1]

df['data'] = an_txt + '-' + luna_txt.map(luna_cod) + '-15'
df['An'] = an_txt.astype(int)
df['Trim'] = luna_txt.map(luna_trim)

# ---- 5. curatare HPI si IRCC (la fel ca in 01_EDA.py) ----------------------

hpi.columns = hpi.columns.str.strip()
hpi = hpi[hpi.iloc[:, 0].astype(str).str.match(r'^\d{4}$')].copy()
hpi.columns.values[0] = 'An'
hpi.columns.values[1] = 'Trim'
hpi.columns.values[2] = 'HPI_Total'
hpi['An'] = hpi['An'].astype(int)
hpi['Trim'] = hpi['Trim'].astype(str).str.strip()
hpi['HPI_Total'] = pd.to_numeric(hpi['HPI_Total'], errors='coerce')
hpi = hpi[['An', 'Trim', 'HPI_Total']].dropna()

ircc.columns = ircc.columns.str.strip()
ircc = ircc[ircc.iloc[:, 0].notna()].copy()
ircc.columns.values[0] = 'Trim_Ref'
ircc.columns.values[2] = 'IRCC_pct'
ircc['IRCC_pct'] = pd.to_numeric(ircc['IRCC_pct'], errors='coerce')
ircc = ircc.dropna(subset=['IRCC_pct'])
# din "T1 2019" scoatem trimestrul si anul separat
ircc_split = ircc['Trim_Ref'].str.extract(r'(T\d)\s*(\d{4})')
ircc['Trim'] = ircc_split[0]
ircc['An'] = ircc_split[1].astype(int)
ircc = ircc[['An', 'Trim', 'IRCC_pct']]

# ---- 6. imbinare (merge) cu datele imobiliare -------------------------------

df = df.merge(hpi, on=['An', 'Trim'], how='left')
df = df.rename(columns={'HPI_Total': 'HPI_trim'})

df = df.merge(ircc, on=['An', 'Trim'], how='left')
df = df.rename(columns={'IRCC_pct': 'IRCC_trim_pct'})
df['IRCC_trim_pct'] = df['IRCC_trim_pct'] * 100  # era exprimat ca fractie (0.0586 -> 5.86)

# autorizatii - cautam manual dupa (judet, an), pentru ca aut are cate un rand per judet/an
aut_lookup = aut.set_index(['judet', 'an'])['autorizatii_rezidentiale']
autorizatii = []
for _, rand in df.iterrows():
    cheie = (rand['judet'], rand['An'])
    if cheie in aut_lookup.index:
        autorizatii.append(aut_lookup.loc[cheie])
    else:
        autorizatii.append(np.nan)
df['autorizatii_judet_an'] = autorizatii

# ---- 7. variabile derivate ---------------------------------------------------

df['pret_mp_log'] = np.log(df['pret_eur_mp'])
df['hpi_norm'] = df['HPI_trim'] / 100
df['sursa'] = 'imobiliare_scrape'

df.insert(0, 'id', range(1, len(df) + 1))

coloane_finale = ['id', 'data', 'oras', 'judet', 'regiune', 'tip_prop', 'camere', 'suprafata_mp',
                   'pret_euro', 'pret_eur_mp', 'an_constructie', 'zona', 'HPI_trim', 'IRCC_trim_pct',
                   'autorizatii_judet_an', 'pret_mp_log', 'hpi_norm', 'sursa']
df_final = df[coloane_finale]

# cate valori lipsa au ramas dupa merge (util de verificat inainte de export)
print('valori lipsa pe coloana:')
print(df_final.isnull().sum())

df_final.to_csv('data/processed/dataset_final.csv', index=False)
print(f"\nsalvat data/processed/dataset_final.csv cu {len(df_final)} randuri")
