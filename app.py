import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objs as go
from datetime import datetime
import io

# Configuration de la page
st.set_page_config(
    page_title="Prévision des Ventes - Saisonnalité & Tendances Clients",
    page_icon="📊",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-title {color: #1f77b4; text-align: center; font-size: 2.5rem;}
    .section-header {color: #2ca02c; border-bottom: 2px solid #eee; padding: 0.5rem 0;}
    .positive {color: #00cc00;}
    .negative {color: #ff0000;}
    .metric-box {background-color: #f9f9f9; border-radius: 10px; padding: 15px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-title">📈 Prévision des Ventes - Saisonnalité & Tendances Clients</h1>', unsafe_allow_html=True)

# Téléchargement des données
st.sidebar.header("1. Chargement des Données")
uploaded_file = st.sidebar.file_uploader(
    "Téléverser un fichier CSV/Excel",
    type=["csv", "xlsx"],
    help="Colonnes requises : date (format JJ/MM/AAAA), quantite_vente"
)

# Paramètres de prévision
st.sidebar.header("2. Paramètres de Prévision")
periods = st.sidebar.slider(
    "Périodes futures à prévoir (jours)",
    min_value=7,
    max_value=365,
    value=90,
    help="Nombre de jours dans le futur pour la prévision"
)

seasonality_mode = st.sidebar.selectbox(
    "Mode de Saisonnalité",
    ["additive", "multiplicative"],
    index=1,
    help="Modèle additif ou multiplicatif pour les variations saisonnières"
)

confidence_interval = st.sidebar.slider(
    "Intervalle de Confiance",
    min_value=0.80,
    max_value=0.99,
    value=0.95,
    step=0.01,
    help="Niveau de certitude des prévisions"
)

# Fonction pour charger les données
def load_data(uploaded_file):
    if uploaded_file is None:
        # Charger le fichier par défaut depuis le répertoire
        try:
            df = pd.read_csv("database.csv", sep=";")
            st.info("Chargement du fichier par défaut database.csv")
            return df
        except:
            st.warning("Fichier par défaut non trouvé. Veuillez télécharger un fichier.")
            return None
    
    try:
        if uploaded_file.name.endswith('.csv'):
            # Essayer plusieurs séparateurs
            content = uploaded_file.getvalue().decode('utf-8')
            try:
                df = pd.read_csv(io.StringIO(content), sep=";")
            except:
                df = pd.read_csv(io.StringIO(content))
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Erreur de lecture du fichier: {str(e)}")
        return None

# Traitement des données
def process_data(df):
    if df is None:
        return None
        
    # Créer une copie pour éviter les modifications sur l'original
    df = df.copy()
    
    # Renommer les colonnes pour standardisation
    column_mapping = {
        'date': 'Date',
        'quantite_vente': 'Ventes'
    }
    
    # Renommer les colonnes si elles existent
    for original, new in column_mapping.items():
        if original in df.columns:
            df.rename(columns={original: new}, inplace=True)
    
    # Vérifier les colonnes nécessaires
    if 'Date' not in df.columns:
        st.error("Erreur : Aucune colonne de date trouvée. Veuillez vérifier votre fichier.")
        return None
    
    if 'Ventes' not in df.columns:
        st.error("Erreur : Aucune colonne de ventes trouvée. Veuillez vérifier votre fichier.")
        return None
    
    # Conversion des dates avec format jour/mois/année
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    
    # Supprimer les lignes avec dates invalides
    df = df.dropna(subset=['Date'])
    
    # Trier par date
    df = df.sort_values('Date')
    
    # Agrégation par jour (somme des ventes quotidiennes)
    df = df.groupby('Date')['Ventes'].sum().reset_index()
    
    return df[['Date', 'Ventes']].rename(columns={'Date': 'ds', 'Ventes': 'y'})

# Charger les données
df = load_data(uploaded_file)
if df is None:
    st.stop()

# Traitement des données
processed_df = process_data(df)
if processed_df is None:
    st.stop()

# Affichage des données brutes
st.header("🔍 Exploration des Données")
st.markdown(f"**Données du {processed_df['ds'].min().strftime('%d/%m/%Y')} au {processed_df['ds'].max().strftime('%d/%m/%Y')}**")
st.dataframe(processed_df.head(10), height=300)

# Statistiques descriptives
st.subheader("Statistiques Descriptives")
stats = processed_df.describe().T
stats['variance'] = processed_df.var()
st.dataframe(stats.style.format("{:.2f}"))

# Analyse temporelle
st.subheader("Évolution Historique des Ventes")
fig_raw = go.Figure()
fig_raw.add_trace(go.Scatter(
    x=processed_df['ds'], 
    y=processed_df['y'], 
    mode='lines+markers',
    name='Ventes réelles',
    line=dict(color='#1f77b4', width=2)
))
fig_raw.update_layout(
    xaxis_title='Date',
    yaxis_title='Volume de Ventes',
    hovermode='x unified',
    template='plotly_white'
)
st.plotly_chart(fig_raw, use_container_width=True)

# Entraînement du modèle Prophet
st.header("⚙️ Modélisation des Prévisions")
st.markdown("**Configuration du modèle Prophet**")
with st.spinner('Entraînement du modèle en cours...'):
    model = Prophet(
        seasonality_mode=seasonality_mode,
        yearly_seasonity=True,
        weekly_seasonity=True,
        daily_seasonity=False,
        interval_width=confidence_interval
    )
    model.add_country_holidays(country_name='FR')
    
    try:
        model.fit(processed_df)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        st.success("Modèle entraîné avec succès!")
    except Exception as e:
        st.error(f"Erreur lors de l'entraînement: {str(e)}")
        st.stop()

# Affichage des prévisions
st.subheader("📊 Résultats des Prévisions")
last_date = processed_df['ds'].max()
forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
forecast_df['Type'] = ['Historique' if d <= last_date else 'Prévision' for d in forecast_df['ds']]

# Graphique interactif des prévisions
fig_forecast = plot_plotly(model, forecast, xlabel='Date', ylabel='Ventes')
fig_forecast.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    hovermode='x unified',
    title="Prévision des Ventes avec Intervalles de Confiance"
)
st.plotly_chart(fig_forecast, use_container_width=True)

# Composantes du modèle
st.subheader("🧠 Analyse des Composantes du Modèle")
st.markdown("""
- **Tendance**: Évolution générale des ventes
- **Saisonnalité hebdomadaire**: Variations récurrentes chaque semaine
- **Saisonnalité annuelle**: Variations récurrentes chaque année
- **Jours fériés**: Impact des jours fériés sur les ventes
""")
fig_components = plot_components_plotly(model, forecast)
st.plotly_chart(fig_components, use_container_width=True)

# Métriques de performance
st.subheader("📈 Performance du Modèle")
if len(processed_df) > 30:
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
    
    # Validation sur les 30 derniers jours
    test_size = min(30, len(processed_df) // 3)
    test_df = processed_df.iloc[-test_size:]
    preds = forecast.iloc[-test_size - periods:-periods]['yhat'].values
    
    mae = mean_absolute_error(test_df['y'], preds)
    mape = mean_absolute_percentage_error(test_df['y'], preds) * 100
    
    col1, col2 = st.columns(2)
    col1.metric("MAE (Erreur Absolue Moyenne)", f"{mae:.2f}")
    col2.metric("MAPE (Erreur Pourcentage Moyenne)", f"{mape:.2f}%", 
                delta="Bonne performance" if mape < 15 else "Amélioration nécessaire",
                delta_color="normal")
    
    # Graphique de validation
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(
        x=test_df['ds'], y=test_df['y'], 
        name='Ventes Réelles', mode='lines+markers'
    ))
    fig_val.add_trace(go.Scatter(
        x=test_df['ds'], y=preds, 
        name='Prévisions', mode='lines+markers'
    ))
    fig_val.update_layout(
        title='Validation du Modèle (30 derniers jours)',
        xaxis_title='Date',
        yaxis_title='Ventes',
        template='plotly_white'
    )
    st.plotly_chart(fig_val, use_container_width=True)
else:
    st.warning("Données insuffisantes pour l'évaluation du modèle (minimum 30 jours requis)")

# Téléchargement des prévisions
st.subheader("💾 Export des Résultats")
csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
st.download_button(
    label="Télécharger les prévisions (CSV)",
    data=csv,
    file_name=f"previsions_ventes_{datetime.now().strftime('%Y%m%d')}.csv",
    mime='text/csv'
)

# Analyse des tendances clients
st.header("👥 Analyse des Tendances Clients")
st.markdown("""
Cette section permet d'identifier les comportements d'achat récurrents et les opportunités commerciales :
""")

# Détection automatique des pics
if 'y' in processed_df.columns:
    processed_df['month'] = processed_df['ds'].dt.month_name()
    monthly_sales = processed_df.groupby('month')['y'].sum().reset_index()
    
    # Ordre des mois
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    monthly_sales['month'] = pd.Categorical(monthly_sales['month'], categories=month_order, ordered=True)
    monthly_sales = monthly_sales.sort_values('month')
    
    # Identification des meilleurs mois
    best_month = monthly_sales.loc[monthly_sales['y'].idxmax()]
    
    col1, col2 = st.columns(2)
    col1.subheader("Saisonnalité Mensuelle")
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=monthly_sales['month'], 
        y=monthly_sales['y'],
        marker_color=['#2ca02c' if m == best_month['month'] else '#1f77b4' for m in monthly_sales['month']]
    ))
    fig_monthly.update_layout(
        xaxis_title='Mois',
        yaxis_title='Ventes Totales',
        template='plotly_white'
    )
    col1.plotly_chart(fig_monthly, use_container_width=True)
    
    col2.subheader("Recommandations Commerciales")
    col2.markdown(f"""
    **Périodes clés identifiées :**
    - 📈 Meilleur mois : **{best_month['month']}** ({best_month['y']:.0f} ventes)
    - 💡 Période propice pour les promotions
    - 🎯 Ciblage marketing accru
    
    **Stratégies suggérées :**
    - Développer des offres saisonnières
    - Adapter le stockage aux pics de demande
    - Préparer des campagnes marketing 1 mois avant les pics
    """)

# Analyse des produits
st.subheader("📦 Analyse des Produits")
if 'nom_produit' in df.columns:
    top_products = df.groupby('nom_produit')['quantite_vente'].sum().nlargest(10).reset_index()
    
    fig_products = go.Figure()
    fig_products.add_trace(go.Bar(
        x=top_products['quantite_vente'],
        y=top_products['nom_produit'],
        orientation='h',
        marker_color='#ff7f0e'
    ))
    fig_products.update_layout(
        title='Top 10 des Produits les Plus Vendus',
        xaxis_title='Quantité Vendue',
        yaxis_title='Produit',
        template='plotly_white'
    )
    st.plotly_chart(fig_products, use_container_width=True)

# Analyse des segments clients
st.subheader("👤 Analyse des Segments Clients")
if 'segment' in df.columns:
    segment_sales = df.groupby('segment')['quantite_vente'].sum().reset_index()
    
    fig_segment = go.Figure()
    fig_segment.add_trace(go.Pie(
        labels=segment_sales['segment'],
        values=segment_sales['quantite_vente'],
        hole=0.3,
        marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c']
    ))
    fig_segment.update_layout(
        title='Répartition des Ventes par Segment Client',
        template='plotly_white'
    )
    st.plotly_chart(fig_segment, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("📆 Application développée avec Streamlit | Prophet | Plotly")
st.markdown("ℹ️ Les prévisions sont basées sur des modèles statistiques et doivent être interprétées avec d'autres indicateurs métier")
