import matplotlib.pyplot as plt
# filepath: /workspaces/project/app.py

# ...existing code...

# Exemple pour remplacer un graphique Plotly par Matplotlib
st.subheader("📈 Évolution Historique des Ventes (Matplotlib)")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(processed_df['ds'], processed_df['y'], marker='o', color='#1f77b4', label='Ventes réelles')
ax.set_title('Historique des Ventes')
ax.set_xlabel('Date')
ax.set_ylabel('Volume de Ventes')
ax.legend()
ax.grid(True)
st.pyplot(fig)
