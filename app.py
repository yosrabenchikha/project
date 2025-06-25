subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "-r", "requirements.txt"])
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from datetime import datetime
import io
import sys
import logging

# Configuration du logging pour détecter les problèmes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Solution de contournement pour Prophet
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly, plot_components_plotly
except ImportError:
    logger.warning("Prophet n'a pas pu être importé. Tentative d'installation...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "prophet==1.1.5"])
    from prophet import Prophet
    from prophet.plot import plot_plotly, plot_components_plotly

# Configuration de la page
st.set_page_config(
    page_title="Prévision des Ventes - Saisonnalité & Tendances Clients",
    page_icon="📊",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-title {
        color: #1f77b4; 
        text-align: center; 
        font-size: 2.5rem;
        padding: 10px;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    .section-header {
        color: #2ca02c; 
        border-bottom: 2px solid #eee; 
        padding: 0.5rem 0;
        margin-top: 1.5rem;
    }
    .positive {color: #00cc00;}
    .negative {color: #ff0000;}
    .metric-box {
        background-color: #f9f9f9; 
        border-radius: 10px; 
        padding: 15px; 
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #2ca02c !important;
        color: white !important;
        border-radius: 8px;
        padding: 8px 16px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1f77b4 !important;
        transform: scale(1.05);
    }
    footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        background-color: #f0f2f6;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal avec un design amélioré
st.markdown("""
<div style="text-align: center; padding: 20px; background: #f0f2f6; border-radius: 10px; margin-bottom: 30px;">
    <h1 class="main-title">📈 Prévision des Ventes - Saisonnalité & Tendances Clients</h1>
    <p style="color: #555; font-size: 1.1rem;">Analyse avancée des ventes avec détection de tendances et prévisions saisonnières</p>
</div>
""", unsafe_allow_html=True)

# Fonction pour charger les données avec gestion d'erreur améliorée
def load_data(uploaded_file):
    if uploaded_file is None:
        try:
            # Charger le fichier par défaut
            df = pd.read_csv("database.csv", sep=";")
            st.info("Chargement du fichier par défaut 'database.csv'")
            return df
        except Exception as e:
            st.warning(f"Fichier par défaut non trouvé: {str(e)}")
            return None
    
    try:
        # Détection du type de fichier
        if uploaded_file.name.endswith('.csv'):
            # Essayer plusieurs séparateurs
            content = uploaded_file.getvalue().decode('utf-8')
            for sep in [';', ',', '\t']:
                try:
                    df = pd.read_csv(io.StringIO(content), sep=sep)
                    if len(df.columns) > 1:
                        st.success(f"Fichier CSV lu avec séparateur: '{sep}'")
                        return df
                except:
                    continue
            # Si aucun séparateur ne fonctionne
            st.error("Impossible de lire le fichier CSV. Essayez avec un séparateur différent.")
            return None
        
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        
    except Exception as e:
        st.error(f"Erreur de lecture du fichier: {str(e)}")
        return None

# Traitement des données avec validation robuste
def process_data(df):
    if df is None:
        return None
        
    # Créer une copie pour éviter les modifications sur l'original
    df = df.copy()
    
    # Standardiser les noms de colonnes
    column_mapping = {
        'date': 'Date',
        'quantite_vente': 'Ventes',
        'jour': 'Date',
        'sales': 'Ventes'
    }
    
    # Renommer les colonnes
    for original, new in column_mapping.items():
        if original in df.columns:
            df.rename(columns={original: new}, inplace=True)
    
    # Vérifier les colonnes nécessaires
    if 'Date' not in df.columns:
        st.error("Erreur : Aucune colonne de date trouvée. Veuillez vérifier votre fichier.")
        return None
    
    if 'Ventes' not in df.columns:
        # Essayer de trouver une colonne numérique pour les ventes
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) > 0:
            df['Ventes'] = df[numeric_cols[0]]
            st.warning(f"Colonne des ventes déduite: {numeric_cols[0]}")
        else:
            st.error("Erreur : Aucune colonne numérique pour les ventes trouvée.")
            return None
    
    # Conversion des dates avec format jour/mois/année
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    
    # Vérifier les dates invalides
    if df['Date'].isna().any():
        invalid_count = df['Date'].isna().sum()
        st.warning(f"{invalid_count} lignes avec dates invalides seront supprimées")
        df = df.dropna(subset=['Date'])
    
    if df.empty:
        st.error("Aucune donnée valide après nettoyage des dates.")
        return None
    
    # Agrégation par jour (somme des ventes quotidiennes)
    daily_sales = df.groupby('Date')['Ventes'].sum().reset_index()
    
    return daily_sales[['Date', 'Ventes']].rename(columns={'Date': 'ds', 'Ventes': 'y'})

# Interface utilisateur
with st.sidebar:
    st.header("⚙️ Paramètres de l'Application")
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=50)
    st.markdown("### Chargement des données")
    uploaded_file = st.file_uploader(
        "Téléverser un fichier CSV/Excel",
        type=["csv", "xlsx"],
        help="Colonnes requises : date (format JJ/MM/AAAA), ventes"
    )
    
    st.markdown("### Paramètres de Prévision")
    periods = st.slider(
        "Périodes futures à prévoir (jours)",
        min_value=7,
        max_value=365,
        value=90,
        help="Nombre de jours dans le futur pour la prévision"
    )

    seasonality_mode = st.selectbox(
        "Mode de Saisonnalité",
        ["additive", "multiplicative"],
        index=1,
        help="Modèle additif ou multiplicatif pour les variations saisonnières"
    )

    confidence_interval = st.slider(
        "Intervalle de Confiance",
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="Niveau de certitude des prévisions"
    )
    
    st.markdown("---")
    st.markdown("**Aide technique**")
    if st.button("Vérifier les dépendances"):
        try:
            from prophet import Prophet
            st.success("Prophet est correctement installé!")
        except ImportError:
            st.error("Prophet n'est pas installé. Cliquez sur le bouton ci-dessous.")
            
        if st.button("Installer Prophet"):
            with st.spinner("Installation en cours..."):
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "prophet==1.1.5"])
                st.success("Prophet installé avec succès! Veuillez redémarrer l'application.")

# Charger les données
df = load_data(uploaded_file)
if df is None:
    # Créer des données de démonstration si aucun fichier n'est chargé
    st.warning("Utilisation de données de démonstration")
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq='D')
    sales = np.random.poisson(lam=100, size=len(dates)) * np.sin(np.arange(len(dates)) * 0.1) + 200
    df = pd.DataFrame({'Date': dates, 'Ventes': sales})

# Traitement des données
processed_df = process_data(df)
if processed_df is None:
    st.error("Erreur critique dans le traitement des données. Impossible de continuer.")
    st.stop()

# Affichage des données brutes
with st.expander("🔍 Exploration des Données", expanded=True):
    st.markdown(f"**Données du {processed_df['ds'].min().strftime('%d/%m/%Y')} au {processed_df['ds'].max().strftime('%d/%m/%Y')}**")
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.dataframe(processed_df.head(10), height=300)
    
    with col2:
        st.subheader("Statistiques Descriptives")
        stats = processed_df.describe().T
        stats['variance'] = processed_df.var()
        st.dataframe(stats.style.format("{:.2f}"), height=300)

# Analyse temporelle
st.subheader("📈 Évolution Historique des Ventes")
fig_raw = go.Figure()
fig_raw.add_trace(go.Scatter(
    x=processed_df['ds'], 
    y=processed_df['y'], 
    mode='lines+markers',
    name='Ventes réelles',
    line=dict(color='#1f77b4', width=2),
    marker=dict(size=5)
))
fig_raw.update_layout(
    xaxis_title='Date',
    yaxis_title='Volume de Ventes',
    hovermode='x unified',
    template='plotly_white',
    height=500
)
st.plotly_chart(fig_raw, use_container_width=True)

# Entraînement du modèle Prophet
st.header("⚙️ Modélisation des Prévisions")
st.markdown("**Configuration du modèle Prophet**")

with st.spinner('Entraînement du modèle en cours...'):
    try:
        model = Prophet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=confidence_interval
        )
        model.add_country_holidays(country_name='FR')
        
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
    title="Prévision des Ventes avec Intervalles de Confiance",
    height=600
)
st.plotly_chart(fig_forecast, use_container_width=True)

# Composantes du modèle
st.subheader("🧠 Analyse des Composantes du Modèle")
with st.expander("Explication des composantes", expanded=False):
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
    col1.metric("MAE (Erreur Absolue Moyenne)", f"{mae:.2f}", 
                help="Mesure l'erreur moyenne dans les prévisions")
    col2.metric("MAPE (Erreur Pourcentage Moyenne)", f"{mape:.2f}%", 
                delta="Bonne performance" if mape < 15 else "Amélioration nécessaire",
                delta_color="normal",
                help="Mesure l'erreur relative moyenne")
    
    # Graphique de validation
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(
        x=test_df['ds'], y=test_df['y'], 
        name='Ventes Réelles', mode='lines+markers',
        line=dict(color='#1f77b4', width=3)
    ))
    fig_val.add_trace(go.Scatter(
        x=test_df['ds'], y=preds, 
        name='Prévisions', mode='lines+markers',
        line=dict(color='#ff7f0e', width=3, dash='dash')
    ))
    fig_val.update_layout(
        title='Validation du Modèle (30 derniers jours)',
        xaxis_title='Date',
        yaxis_title='Ventes',
        template='plotly_white',
        height=400
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
    mime='text/csv',
    help="Téléchargez les prévisions au format CSV pour une analyse ultérieure"
)

# Analyse des tendances clients
st.header("👥 Analyse des Tendances Clients")
st.markdown("""
Cette section permet d'identifier les comportements d'achat récurrents et les opportunités commerciales :
""")

col1, col2 = st.columns(2)

with col1:
    # Saisonnalité mensuelle
    st.subheader("📅 Saisonnalité Mensuelle")
    processed_df['month'] = processed_df['ds'].dt.month_name()
    monthly_sales = processed_df.groupby('month')['y'].sum().reset_index()
    
    # Ordre des mois
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    monthly_sales['month'] = pd.Categorical(monthly_sales['month'], categories=month_order, ordered=True)
    monthly_sales = monthly_sales.sort_values('month')
    
    # Identification des meilleurs mois
    best_month = monthly_sales.loc[monthly_sales['y'].idxmax()]
    
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=monthly_sales['month'], 
        y=monthly_sales['y'],
        marker_color=['#2ca02c' if m == best_month['month'] else '#1f77b4' for m in monthly_sales['month']],
        text=monthly_sales['y'],
        textposition='auto'
    ))
    fig_monthly.update_layout(
        xaxis_title='Mois',
        yaxis_title='Ventes Totales',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

with col2:
    # Recommandations commerciales
    st.subheader("💡 Recommandations Commerciales")
    st.markdown(f"""
    **Périodes clés identifiées :**
    - 📈 Meilleur mois : **{best_month['month']}** ({best_month['y']:.0f} ventes)
    - 💡 Période propice pour les promotions
    - 🎯 Ciblage marketing accru
    
    **Stratégies suggérées :**
    - Développer des offres saisonnières
    - Adapter le stockage aux pics de demande
    - Préparer des campagnes marketing 1 mois avant les pics
    """)
    
    # Analyse des jours de la semaine
    st.subheader("📆 Analyse Hebdomadaire")
    processed_df['day_of_week'] = processed_df['ds'].dt.day_name()
    daily_sales = processed_df.groupby('day_of_week')['y'].mean().reset_index()
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_sales['day_of_week'] = pd.Categorical(daily_sales['day_of_week'], categories=day_order, ordered=True)
    daily_sales = daily_sales.sort_values('day_of_week')
    
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Bar(
        x=daily_sales['day_of_week'],
        y=daily_sales['y'],
        marker_color='#ff7f0e'
    ))
    fig_daily.update_layout(
        xaxis_title='Jour de la semaine',
        yaxis_title='Ventes Moyennes',
        template='plotly_white',
        height=300
    )
    st.plotly_chart(fig_daily, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<footer>
    <p>📆 Application développée avec Streamlit | Prophet | Plotly</p>
    <p>ℹ️ Les prévisions sont basées sur des modèles statistiques et doivent être interprétées avec d'autres indicateurs métier</p>
</footer>
""", unsafe_allow_html=True)
