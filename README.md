# Projet : Prévision des Ventes Mensuelles

## Objectif du projet

Ce projet a pour objectif d’analyser les données de ventes d’une entreprise afin de :

- Comprendre l’évolution des ventes mensuelles
- Identifier les tendances et comportements
- Construire un modèle simple de prévision
- Aider à la prise de décision business


 ### 1. Préparation des données

Les étapes suivantes ont été réalisées :

- Suppression des doublons
- Traitement des valeurs manquantes
- Standardisation des dates
- Nettoyage des colonnes (formatage texte)
- Vérification de la cohérence des données :
  - recalcul du montant total
  - contrôle des prix unitaires

### 2. Analyse exploratoire (EDA)

L’analyse a permis de répondre aux questions suivantes :

- Quels sont les mois les plus performants ?
- Quels produits génèrent le plus de ventes ?
- Quelle région est la plus rentable ?
- Existe-t-il une saisonnalité ?

#### Visualisations réalisées

- Évolution des ventes par produit
- Comparaison des ventes mensuelles
- Analyse des performances par région
- Graphique prédiction vs réel

## 3. Modélisation

Un modèle de régression linéaire a été utilisé pour prédire les ventes mensuelles.

#### Principe

Le modèle établit une relation entre :

- le temps (mois)
- le montant des ventes

#### Étapes :

- Agrégation des ventes par mois
- Création d’une variable temporelle ("mois_num")
- Entraînement du modèle
- Précision avec R² (R2 Score)

### 4. Résultats

- Les ventes présentent une forte variabilité
- Aucune tendance linéaire claire n’a été identifiée
- Le modèle donne une tendance globale mais reste imprécis

#### Limites

- Dataset limité à une seule année
- Nombre d’observations faible
- Absence de variables explicatives supplémentaires

 ## 5. Recommandations

- Collecter plus de données (plusieurs années)
- Ajouter des variables métier (promotions, saison, stock)
- Analyser les performances par produit et région
- Anticiper les périodes de baisse

## Technologies utilisées

- Python
- Pandas
- Numpy
- Matplotlib
- Scikit-learn


# Le projet inclut :

- Analyse des données
- Visualisations
- Modèle de prédiction
- Rapport final avec recommandations

