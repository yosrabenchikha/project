import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(page_title="Prévision des Ventes", layout="wide", page_icon="📊")
st.title("📈 Tableau de Bord des Prévisions de Ventes")

# Chargement des données
@st.cache_data
def load_data():
     df = pd.read_csv(
    "/content/database.csv",
    sep=";",
    encoding='latin1',         # Encodage pour caractères français

)

df = load_data()

# Sidebar - Contrôles utilisateur
st.sidebar.header("Paramètres de Visualisation")
selected_segment = st.sidebar.selectbox("Segment Client", options=df['segment'].unique())
forecast_months = st.sidebar.slider("Mois de Prévision", 1, 24, 12)
show_confidence = st.sidebar.checkbox("Afficher l'intervalle de confiance", value=True)

# Section 1: Graphique temporel interactif
st.header("Historique des Ventes et Prévisions")
col1, col2 = st.columns([3, 1])

with col1:
    # Filtrage des données
    segment_data = df[df['segment'] == selected_segment]
    
    # Création du graphique
    fig = go.Figure()
    
    # Historique des ventes
    fig.add_trace(go.Scatter(
        x=segment_data['date'],
        y=segment_data['ventes'],
        name='Ventes Réelles',
        line=dict(color='#1f77b4', width=3)
    ))
    
    # Prévisions
    fig.add_trace(go.Scatter(
        x=segment_data['date'],
        y=segment_data['prevision'],
        name='Prévisions',
        line=dict(color='#ff7f0e', width=3, dash='dot')
    ))
    
    if show_confidence:
        # Intervalle de confiance (simulé)
        upper_bound = segment_data['prevision'] * 1.1
        lower_bound = segment_data['prevision'] * 0.9
        fig.add_trace(go.Scatter(
            x=segment_data['date'],
            y=upper_bound,
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=segment_data['date'],
            y=lower_bound,
            fill='tonexty',
            fillcolor='rgba(255, 127, 14, 0.2)',
            line=dict(width=0),
            name='Intervalle de confiance'
        ))
    
    # Personnalisation du graphique
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Volume des Ventes',
        hovermode='x unified',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # KPI clés
    st.metric("Dernières ventes", f"{segment_data['ventes'].iloc[-1]:,} €")
    st.metric("Prévision moyenne", f"{segment_data['prevision'].mean():.0f} €")
    st.metric("Croissance prévue", "+8.2%", delta_color="inverse")
    
    # Sélecteur de période
    period = st.radio("Période d'analyse:", ['Mensuelle', 'Trimestrielle', 'Annuelle'])
    
    # Téléchargement des données
    st.download_button(
        label="Télécharger les données",
        data=segment_data.to_csv(index=False).encode('utf-8'),
        file_name=f"ventes_{selected_segment}.csv",
        mime="text/csv"
    )

# Section 2: Analyse saisonnière
st.header("Analyse Saisonnière")

# Calcul des moyennes saisonnières
df['saison'] = df['date'].dt.month.map({
    1: 'Hiver', 2: 'Hiver', 3: 'Printemps', 
    4: 'Printemps', 5: 'Printemps', 6: 'Été',
    7: 'Été', 8: 'Été', 9: 'Automne',
    10: 'Automne', 11: 'Automne', 12: 'Hiver'
})

seasonal_avg = df.groupby(['saison', 'segment'])['ventes'].mean().reset_index()

# Création du graphique à barres
fig2 = px.bar(
    seasonal_avg, 
    x='saison', 
    y='ventes', 
    color='segment',
    barmode='group',
    category_orders={"saison": ["Hiver", "Printemps", "Été", "Automne"]},
    labels={'ventes': 'Ventes Moyennes', 'saison': 'Saison'},
    height=400
)

# Personnalisation
fig2.update_layout(
    title="Ventes Moyennes par Saison et Segment",
    template='plotly_white'
)
st.plotly_chart(fig2, use_container_width=True)

# Section 3: Comparaison segmentée
st.header("Comparaison par Segment Client")

# Calcul des performances par segment
segment_perf = df.groupby('segment').agg({
    'ventes': ['mean', 'sum'],
    'prevision': 'mean'
}).reset_index()
segment_perf.columns = ['Segment', 'Ventes Moyennes', 'Ventes Totales', 'Prévision Moyenne']
segment_perf['Variation'] = (segment_perf['Prévision Moyenne'] / segment_perf['Ventes Moyennes'] - 1) * 100

# Affichage sous forme de tableau
st.dataframe(
    segment_perf.style
    .format({
        'Ventes Moyennes': '{:,.0f} €',
        'Ventes Totales': '{:,.0f} €',
        'Prévision Moyenne': '{:,.0f} €',
        'Variation': '{:.1f}%'
    })
    .bar(subset=['Variation'], align='mid', color=['#FF9999', '#99FF99'])
    .set_properties(**{'background-color': '#f8f9fa', 'border': '1px solid #dee2e6'}),
    use_container_width=True
)

# Section 4: Analyse détaillée (onglets)
st.header("Analyse Détaillée")
tab1, tab2, tab3 = st.tabs(["Tendance", "Saisonnalité", "Résidus"])

with tab1:
    st.subheader("Décomposition de la Tendance")
    # Simulation d'une décomposition
    trend = np.linspace(100, 500, len(segment_data))
    seasonal = 50 * np.sin(np.linspace(0, 4*np.pi, len(segment_data)))
    resid = np.random.normal(0, 20, len(segment_data))
    
    fig3, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(segment_data['date'], segment_data['ventes'], label='Original')
    axes[0].set_title('Ventes Originales')
    axes[1].plot(segment_data['date'], trend, label='Tendance', color='green')
    axes[1].set_title('Composante de Tendance')
    axes[2].plot(segment_data['date'], seasonal, label='Saisonnalité', color='red')
    axes[2].set_title('Composante Saisonnière')
    axes[3].plot(segment_data['date'], resid, label='Résidus', color='purple')
    axes[3].set_title('Résidus')
    plt.tight_layout()
    st.pyplot(fig3)

with tab2:
    st.subheader("Analyse de Saisonnalité")
    # Heatmap saisonnière
    df['mois'] = df['date'].dt.month_name()
    df['annee'] = df['date'].dt.year
    heatmap_data = df.pivot_table(index='mois', columns='annee', values='ventes', aggfunc='sum')
    
    fig4 = px.imshow(
        heatmap_data,
        labels=dict(x="Année", y="Mois", color="Ventes"),
        aspect="auto",
        color_continuous_scale='Viridis'
    )
    fig4.update_layout(title="Heatmap des Ventes par Mois et Année")
    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.subheader("Analyse des Résidus")
    # Simulation de résidus
    resid = np.random.normal(0, 50, len(segment_data))
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribution des Résidus**")
        fig5 = px.histogram(x=resid, nbins=30, labels={'x': 'Résidus'})
        st.plotly_chart(fig5, use_container_width=True)
    
    with col2:
        st.write("**QQ-Plot des Résidus**")
        # QQ-Plot simplifié
        fig6, ax = plt.subplots(figsize=(6, 4))
        stats.probplot(resid, dist="norm", plot=ax)
        ax.set_title('QQ-Plot des Résidus')
        st.pyplot(fig6)

# Section 5: Téléchargement du rapport
st.divider()
st.header("Exporter les Résultats")
report_format = st.selectbox("Format du rapport", ["PDF", "HTML", "PPTX"])
if st.button("Générer le Rapport Complet"):
    with st.spinner("Génération du rapport..."):
        time.sleep(2)
        st.success("Rapport généré avec succès!")
        st.download_button(
            "Télécharger le rapport",
            data="Contenu simulé du rapport".encode('utf-8'),
            file_name=f"rapport_ventes_{datetime.now().strftime('%Y%m%d')}.{report_format.lower()}",
            mime="application/octet-stream"
        )