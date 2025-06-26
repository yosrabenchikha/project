import streamlit as st
import pandas as pd
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Analyse des Ventes",
    page_icon="📊",
    layout="wide"
)

# Titre de l'application
st.title("📈 Analyse Historique des Ventes")
st.subheader("Visualisation des données de vente avec Streamlit")

# Chargement des données avec gestion d'encodage robuste
@st.cache_data
def load_data():
    # Liste des encodages à essayer (les plus courants pour les fichiers français)
    encodings = ['latin1', 'ISO-8859-1', 'cp1252', 'utf-8']
    
    for encoding in encodings:
        try:
            # Lecture du CSV avec séparateur ';'
            df = pd.read_csv(
                'database.csv', 
                sep=';', 
                parse_dates=['date'], 
                dayfirst=True,
                encoding=encoding
            )
            st.sidebar.success(f"Fichier chargé avec l'encodage: {encoding}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            st.error(f"Erreur avec l'encodage {encoding}: {str(e)}")
    
    st.error("Aucun encodage n'a fonctionné. Essayez de convertir votre fichier en UTF-8.")
    return pd.DataFrame()

# Charger les données
df = load_data()

# Si le DataFrame est vide, arrêter l'application
if df.empty:
    st.warning("Le fichier de données n'a pas pu être chargé. Veuillez vérifier le format du fichier.")
    st.stop()

# Nettoyage des colonnes numériques
try:
    numeric_cols = ['quantite_vente', 'prix', 'remise_pct', 'taux-fidelite']
    for col in numeric_cols:
        if col in df.columns:
            # Remplacer les virgules par des points et convertir en numérique
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '.'), 
                errors='coerce'
            )
    
    # Calcul du chiffre d'affaires
    df['chiffre_affaires'] = df['quantite_vente'] * df['prix'] * (1 - df['remise_pct']/100)
except Exception as e:
    st.error(f"Erreur dans le traitement des données: {str(e)}")

# Sidebar pour les filtres
with st.sidebar:
    st.header("Filtres")
    
    # Sélection de la période
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    selected_dates = st.date_input(
        "Période d'analyse",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Sélection des produits
    all_products = df['nom_produit'].unique()
    selected_products = st.multiselect(
        "Produits à inclure",
        options=all_products,
        default=all_products[:5] if len(all_products) > 5 else all_products
    )
    
    # Sélection des segments clients
    segments = df['segment'].unique()
    selected_segments = st.multiselect(
        "Segments clients",
        options=segments,
        default=segments
    )

# Application des filtres
if len(selected_dates) == 2:
    start_date, end_date = selected_dates
    filtered_df = df[
        (df['date'].dt.date >= start_date) &
        (df['date'].dt.date <= end_date) &
        (df['nom_produit'].isin(selected_products)) &
        (df['segment'].isin(selected_segments))
    ]
else:
    filtered_df = df

if filtered_df.empty:
    st.warning("Aucune donnée disponible avec les filtres sélectionnés")
else:
    # Agrégation des données par jour
    daily_sales = filtered_df.groupby('date').agg({
        'quantite_vente': 'sum',
        'chiffre_affaires': 'sum'
    }).reset_index()
    
    # Métriques
    col1, col2, col3 = st.columns(3)
    col1.metric("Période analysée", f"{start_date} - {end_date}")
    col2.metric("Ventes totales", f"{daily_sales['quantite_vente'].sum():,d} unités")
    col3.metric("Chiffre d'affaires", f"{daily_sales['chiffre_affaires'].sum():,.0f} €")
    
    # Graphiques
    st.subheader("Évolution Journalière des Ventes")
    
    tab1, tab2 = st.tabs(["Quantités Vendues", "Chiffre d'Affaires"])
    
    with tab1:
        st.area_chart(
            daily_sales.set_index('date')['quantite_vente'],
            use_container_width=True,
            height=400
        )
        
    with tab2:
        st.area_chart(
            daily_sales.set_index('date')['chiffre_affaires'],
            use_container_width=True,
            height=400
        )
    
    # Top produits
    st.subheader("Top 10 des Produits")
    top_products = filtered_df.groupby('nom_produit').agg({
        'quantite_vente': 'sum',
        'chiffre_affaires': 'sum'
    }).nlargest(10, 'chiffre_affaires').reset_index()
    
    st.dataframe(
        top_products.style.format({
            'quantite_vente': '{:,.0f}',
            'chiffre_affaires': '{:,.0f} €'
        }),
        use_container_width=True
    )
    
    # Données brutes
    if st.checkbox("Afficher les données brutes"):
        st.subheader("Données Brutes")
        st.dataframe(filtered_df, use_container_width=True)

# Pied de page
st.caption(f"Application développée avec Streamlit • {len(df)} lignes chargées • Données mises à jour le {datetime.now().strftime('%d/%m/%Y %H:%M')}")
