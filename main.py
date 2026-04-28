import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Cleaning
df = pd.read_csv('donnees_ventes_mensuelles.csv', sep=';')
df = df.drop_duplicates()
df['date_vente'] = pd.to_datetime(df['date_vente'], format='%d/%m/%Y')
df = df.dropna()
cols_nettoyer = ['commercial', 'produit', 'region', 'categorie']
df[cols_nettoyer] = df[cols_nettoyer].apply(lambda x: x.str.strip().str.capitalize())

###Verification des coherences

##Coherence du montant total
df['montant_attendu'] = (df['quantite'] * df['prix_unitaire']).round(2)

# 2. Identification des lignes incoherentes
# On cree un nouveau DataFrame qui contient uniquement les erreurs
incoherences = df[df['montant_total'] != df['montant_attendu']]

# 3. Affichage du diagnostic
print(f"--- Diagnostic de Coherence ---")
print(f"Nombre total de lignes analysees : {len(df)}")
print(f"Nombre d'erreurs de calcul detectees : {len(incoherences)}")

if len(incoherences) > 0:
    print("\nAperçu des erreurs trouvees :")
    # On affiche les colonnes cles pour comparer l'erreur et la correction
    print(incoherences[['produit', 'quantite', 'prix_unitaire', 'montant_total', 'montant_attendu']].head())
    
    # 4. Correction des donnees
    # On remplace les valeurs fausses par les valeurs calculees
    df['montant_total'] = df['montant_attendu']
    print("\nCorrection appliquee : La colonne 'montant_total' a ete mise à jour.")
else:
    print("\nAucune incoherence detectee. Les calculs sont exacts.")

# 5. Nettoyage final
# On supprime la colonne temporaire 'montant_attendu' pour garder le DataFrame propre
df.drop(columns=['montant_attendu'], inplace=True)

#Ventes actuelles par produit

df['vente_mens'] = df['date_vente'].dt.to_period('M')
pivot = pd.pivot_table(
    df,
    values="montant_total",
    index="vente_mens",
    columns="produit",
    aggfunc="sum",
    fill_value=0
)
print(f'\n{pivot}')

pivot.plot(figsize=(10,5))
plt.title("Ventes actuelles par produit", fontsize=12, fontweight='bold')
plt.show()


##Coherence des prix unitaires
# 1. Identifier les produits qui ont plus d'un prix unitaire different

prix_par_produit = df.groupby('produit')['prix_unitaire'].nunique()
produits_suspects = prix_par_produit[prix_par_produit > 1]

print("--- Diagnostic de Coherence des Prix ---")

if not produits_suspects.empty:
    print(f"Attention : {len(produits_suspects)} produit(s) ont des prix unitaires variables.")
    
    # 2. Afficher le detail des prix pour ces produits
    for produit in produits_suspects.index:
        prix_list = df[df['produit'] == produit]['prix_unitaire'].unique()
        print(f" - {produit} est vendu a : {prix_list}")
    
   
else:
    print("[OK] Coherence parfaite : chaque produit a un prix unique dans tout le fichier.")

# 4.valeurs aberrantes
prix_aberrants = df[df['prix_unitaire'] <= 0]
if not prix_aberrants.empty:
    print(f"ERREUR : {len(prix_aberrants)} ligne(s) avec un prix negatif ou nul !")
    print(prix_aberrants[['produit', 'prix_unitaire']])

print("\n--- Diagnostic des Valeurs Aberrantes ---")
description = df.describe()

prix_min = description.loc['min', 'prix_unitaire']
prix_max = description.loc['max', 'prix_unitaire']

print(f"Le prix le plus bas est : {prix_min}")
print(f"Le prix le plus haut est : {prix_max}")
if description.loc['min', 'prix_unitaire'] <= 0:
    print("ALERTE : Il y a des prix incoherents (nuls ou negatifs) !")
else:
    print("Aucun prix incoherent detectee.")

### Exploiratory Data Analysis
## Les mois les plus performants
conv_mois = df['date_vente'].dt.strftime('%B') 
mois_perf = df.groupby(conv_mois)['montant_total'].sum().sort_values(ascending=True).head(5)

print(f'\nles mois les plus performents :{mois_perf}')

plt.figure(figsize=(10, 5))
plt.plot(mois_perf.index, mois_perf.values, marker='o', linewidth=2, markersize=8, color='#3498db')
plt.title('Top 5 Mois les Plus Performants', fontsize=12, fontweight='bold')
plt.xlabel('Mois')
plt.ylabel('Montant Total (€)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
 
## Produits les plus rentables
top_produit = df.groupby('produit')['montant_total'].sum()
top_produit = top_produit.sort_values(ascending = False).head(5)

print(f'\nLes produits les plus rentables : {top_produit}')

plt.figure(figsize=(10, 5))
plt.bar(top_produit.index, top_produit.values, color='#2ecc71', edgecolor='black', linewidth=1.5)
plt.title('Top 5 Produits les Plus Rentables', fontsize=12, fontweight='bold')
plt.xlabel('Produit')
plt.ylabel('Montant Total (€)')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

## Regions les plus rentables
top_regions = df.groupby('region')['montant_total'].sum()
top_regions = top_regions.sort_values(ascending = False).head(4)

print(f'\nLes top regions : {top_regions}')

plt.figure(figsize=(10, 5))
plt.bar(top_regions.index, top_regions.values, color='#e74c3c', edgecolor='black', linewidth=1.5)
plt.title('Top Region le Plus Rentable', fontsize=12, fontweight='bold')
plt.xlabel('Region')
plt.ylabel('Montant Total (€)')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()


# Groupement par mois
df['mois_timestamp'] = df['date_vente'].dt.to_period('M').dt.to_timestamp()
vente_mensuelle = df.groupby('mois_timestamp')['montant_total'].sum().reset_index()
print(f'\nVentes mensuelles :\n{vente_mensuelle}')

# 2. Preparation des variables pour le modele
vente_mensuelle['t'] = np.arange(len(vente_mensuelle)) # colonne de tendance temporelle
dummies = pd.get_dummies(vente_mensuelle['mois_timestamp'].dt.month, prefix='m') #colonnes en binaire pour les mois

df_final = pd.concat([vente_mensuelle, dummies], axis=1)

# 3. Modèle
cols_mois = [f'm_{i}' for i in range(1, 13)]
features = ['t'] + cols_mois

X = df_final[features]
y = df_final['montant_total']

model = LinearRegression()
model.fit(X, y)

y_pred_historique = model.predict(X)

# Calcul de la precision (R²)
from sklearn.metrics import r2_score
r2 = r2_score(y, y_pred_historique)
precision_pct = r2 * 100
print(f"Precision du modèle : {precision_pct:.1f}%")

# 4. Prediction Futur
derniere_date = vente_mensuelle['mois_timestamp'].max()
predictions_futures = []

for i in range(1, 2):  
    prochaine_date = derniere_date + pd.DateOffset(months=i)
    prochain_mois_num = prochaine_date.month
    #correspondance entre le mois et les variables dummies(position)
    data_next = {'t': [len(vente_mensuelle) + i - 1]}

   
    for m in range(1, 13):
        data_next[f'm_{m}'] = [1 if m == prochain_mois_num else 0]
    
    X_next = pd.DataFrame(data_next)  #trasnforme en dataframe pour le modele
    prediction = model.predict(X_next[features])[0]
    predictions_futures.append({'date': prochaine_date, 'prediction': prediction})
    
    print(f"Prediction pour {prochaine_date.strftime('%B %Y')} : {prediction:.2f} €")

# Creer un dataframe pour les predictions futures
df_predictions = pd.DataFrame(predictions_futures)
print(df_predictions)


# 5. Graphique
plt.figure(figsize=(12, 6))

# Tracer l'historique reel
plt.plot(vente_mensuelle['mois_timestamp'], y, marker='o', label='Ventes Reelles (2025)', color='#2c3e50', linewidth=2)

# Tracer ce que le modèle a "compris" de l'historique
plt.plot(vente_mensuelle['mois_timestamp'], y_pred_historique, '--', label='Ajustement Modèle', color='#e67e22', alpha=0.7)

# Tracer les predictions future avec une ligne
plt.plot(df_predictions['date'], df_predictions['prediction'], color='red', linewidth=2, marker='o', label='Predictions Futures', zorder=5)
for idx, row in df_predictions.iterrows():
    plt.annotate(f'{row["prediction"]:.0f} €', (row['date'], row['prediction']), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

# Esthetique
plt.title('Analyse et Projection des Ventes Mensuelles', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Montant Total (€)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()