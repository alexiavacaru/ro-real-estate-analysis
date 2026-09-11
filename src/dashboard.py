
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from models import regresie_preturi, prognoza_oras, clustere_kmeans, FEATURES_CLUSTER

st.set_page_config(page_title='Piata imobiliara RO - Dashboard', layout='wide')

dataset = pd.read_csv('data/processed/dataset_final.csv')
serii = pd.read_csv('data/processed/serii_timp_pret_mp.csv')

st.title('Piata imobiliara din Romania - Dashboard')
st.write('Proiect de analiza date: EDA, regresie, serii de timp (ARIMA) si clustering K-Means.')

tab1, tab2, tab3, tab4 = st.tabs(['Date', 'Regresie', 'Serii de timp', 'Clustering'])

# --- tab 1: explorare date -------------------------------------------------
with tab1:
    st.subheader('Preturi €/mp pe orase')

    orase_alese = st.multiselect('Orase', sorted(dataset['oras'].unique()),
                                  default=sorted(dataset['oras'].unique()))
    df_filtrat = dataset[dataset['oras'].isin(orase_alese)]

    fig, ax = plt.subplots(figsize=(8, 4))
    df_filtrat.boxplot(column='pret_eur_mp', by='oras', ax=ax, rot=45)
    plt.suptitle('')
    ax.set_title('Distributie pret €/mp')
    st.pyplot(fig)

    st.write(df_filtrat.groupby('oras')['pret_eur_mp'].agg(['mean', 'median', 'std', 'count']).round(0))
    st.dataframe(df_filtrat)

# --- tab 2: regresie --------------------------------------------------------
with tab2:
    st.subheader('Regresie multipla - predictie pret €/mp')

    rez = regresie_preturi(dataset)
    st.write(rez['rezultate'])

    fig, ax = plt.subplots(figsize=(7, 4))
    coef = rez['coef_ols'].sort_values('Coeficient')
    culori = ['#C00000' if c < 0 else '#1F4E79' for c in coef['Coeficient']]
    ax.barh(coef['Feature'], coef['Coeficient'], color=culori)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title('Coeficienti OLS standardizati')
    st.pyplot(fig)

# --- tab 3: serii de timp ----------------------------------------------------
with tab3:
    st.subheader('Prognoza ARIMA(1,1,1)')

    oras_ales = st.selectbox('Oras', sorted(serii['oras'].unique()))
    pasi = st.slider('Trimestre de prognozat', 1, 8, 4)

    sub, model, forecast = prognoza_oras(serii, oras_ales, pasi)

    fig, ax = plt.subplots(figsize=(9, 4))
    t = range(len(sub))
    ax.plot(t, sub['pret_mediu_eur_mp'], 'o-', label='date reale')
    ax.plot(t, model.fitted, '--', alpha=0.7, label='fitted')
    t_prog = range(len(sub), len(sub) + pasi)
    ax.plot(t_prog, forecast, 's--', color='#C00000', label='prognoza')
    ax.legend()
    ax.set_title(f'{oras_ales} - ARIMA(1,1,1)')
    st.pyplot(fig)

    st.write(f"phi = {model.phi:.3f} | theta = {model.theta:.3f} | AIC = {model.aic():.2f}")

# --- tab 4: clustering -------------------------------------------------------
with tab4:
    st.subheader('Segmentare orase - K-Means')

    k = st.slider('Numar clustere (K)', 2, 6, 3)
    cluster_df, kmeans, metrici = clustere_kmeans(k=k)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        for cl in sorted(cluster_df['cluster'].unique()):
            sub_cl = cluster_df[cluster_df['cluster'] == cl]
            ax.scatter(sub_cl['PC1'], sub_cl['PC2'], label=f'Cluster {cl}', s=140)
            for _, r in sub_cl.iterrows():
                ax.annotate(r['oras'], (r['PC1'], r['PC2']), fontsize=8)
        ax.legend()
        ax.set_title(f'K-Means (K={k}) - PCA 2D')
        st.pyplot(fig)

    with col2:
        st.write('Metrici:')
        for nume, val in metrici.items():
            st.metric(nume, round(val, 3))

    st.dataframe(cluster_df[['oras', 'cluster'] + FEATURES_CLUSTER])
