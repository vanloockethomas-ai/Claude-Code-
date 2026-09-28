## Le projet

SaaS destiné aux PME e-commerce (Shopify ou autre ; secteurs : électronique, maison, accessoires, beauté).

**Proposition de valeur :** un système d'optimisation tarifaire qui recommande automatiquement le prix maximisant la marge, en respectant les contraintes commerciales de l'entreprise cliente.

Bénéfices visés : augmentation de la marge, surveillance des concurrents, recommandations explicables, garde-fous (marge minimale, variation maximale).

## Architecture (à respecter)

Le système est composé de deux briques **séparées** :

1. **Modèle de demande (ML)** : prédit la quantité vendue Q(p) d'un produit pour un prix p donné.
2. **Optimiseur sous contraintes** : teste des prix candidats et choisit celui qui maximise `(p - coût_unitaire) * Q(p)`, en excluant tout prix qui viole un garde-fou.

### Garde-fous, paramétrables par client et par produit

- `target_margin` : objectif de marge unitaire (en %) — **contrainte stricte**
- `max_daily_price_change` : variation maximale par rapport au prix actuel (en %)
- `repricing_frequency` : fréquence de repricing
- `min_price` / `max_price` : prix plancher et prix plafond
- **Jamais de prix sous le coût** (`cost`) : contrainte stricte (vente à perte en principe interdite en Belgique — exceptions à vérifier)
- **Terminaisons de prix** : arrondir le prix recommandé à une terminaison commerciale (ex. ,99 / ,95), paramétrable

Chaque recommandation doit être accompagnée d'une **explication lisible en français** (pourquoi ce prix, quelle contrainte a été active).

### Marge nette

- `cost` = coût unitaire **complet** : achat + livraison + frais de paiement + coût moyen des retours.
- Tous les calculs de marge se font **hors TVA** ; les prix affichés aux clients finaux sont TTC → convertir proprement.

## Stack

- Python 3.11+
- pandas, numpy
- scikit-learn / LightGBM (modèle de demande)
- SciPy ou recherche sur grille (optimisation)
- pytest (tests)
- Streamlit (démo) — plus tard : FastAPI (API), intégration Shopify

## Structure du dépôt

```
data/          # données brutes et synthétiques (jamais de données clients sur Git)
src/
  data/        # chargement, nettoyage, générateur synthétique
  demand/      # modèle de demande
  optimizer/   # optimisation sous contraintes + garde-fous
  explain/     # génération des explications
  backtest/    # simulation sur l'historique
app/           # démo Streamlit
tests/
```

## Feuille de route

- [ ] Étape 1 — Générateur de données synthétiques avec élasticité prix connue
- [ ] Étape 2 — Modèle de demande (doit retrouver l'élasticité synthétique et battre la baseline)
- [ ] Étape 3 — Optimiseur avec garde-fous
- [ ] Étape 4 — Explications des recommandations
- [ ] Étape 5 — Backtest (mesurer le gain de marge)
- [ ] Étape 6 — Démo Streamlit (upload CSV → recommandations)
- [ ] Plus tard — API, intégration Shopify, surveillance des prix concurrents
- [ ] V2 — Démarrage à froid (nouveaux produits : modèle par catégorie), effets entre produits (cannibalisation), niveau de confiance par recommandation, journal des recommandations + réentraînement, contrôle qualité automatique des imports, conformité directive Omnibus pour les promotions

## Format des données d'entrée (Features, x)

Une ligne par produit et par jour :
`date, product_id, category, brand, cost, rating, current_price, competitor_min_price, price_gap_vs_competitor, stock, stock_coverage, is_promotion (0/1), day_of_week, month, is_black_friday (0/1), days_to_black_friday, is_christmas_period (0/1), is_holiday (0/1), views_7d, sales_7d`

- `date` et `product_id` sont des **identifiants**, pas des features du modèle : `date` sert au découpage temporel et à calculer `day_of_week`, `month`, etc.
- `price_gap_vs_competitor` = écart relatif entre `current_price` et `competitor_min_price`.

#### Features dépendantes du prix

L'optimiseur fait varier `current_price`. Pour **chaque prix candidat**, toutes les features qui dépendent du prix (ex. `price_gap_vs_competitor`) doivent être **recalculées** avant la prédiction. Centraliser ce calcul dans une seule fonction utilisée à l'entraînement ET à l'optimisation.

#### Attention aux fuites de données

`sales_7d`, `views_7d`, `stock_coverage` : ces features ne doivent utiliser que les jours passés, en excluant le jour prédit. Sinon, le modèle « voit » la réponse et paraîtra excellent en test mais inutile en réalité.

## Format des données de sortie (Target, y)

Une ligne par produit et par jour :
`units_sold`

#### Ruptures de stock (demande censurée)

Un jour où le stock est tombé à zéro, `units_sold` mesure le stock disponible, pas la vraie demande. Ces jours doivent être **repérés** (ajouter une colonne `is_stockout`) et traités à part (exclus de l'entraînement ou modélisés explicitement) — proposer une approche avant de coder.

## Règles de travail

- **Toujours proposer un plan avant de coder**, et attendre ma validation.
- Travailler **une étape à la fois**, en petits changements.
- **Expliquer chaque choix technique et chaque bloc de code** : je suis en train d'apprendre le machine learning.
- Écrire des tests pytest pour toute nouvelle fonction ; lancer les tests avant de dire qu'une tâche est terminée.
- Code et commentaires en français, noms de variables clairs mais peuvent être en anglais.
- Évaluation des modèles : **découpage temporel** (entraînement sur le passé, test sur une période postérieure), jamais de mélange aléatoire.
- Ne jamais committer de données clients ni de secrets (fichier `.env`, `.gitignore` à jour).
- Mode **recommandation uniquement** : le système ne modifie jamais un prix lui-même.

## Évaluation

- **Baseline obligatoire** : le modèle doit battre une règle simple (ex. ventes = moyenne des 7 derniers jours) ; sinon, il n'apporte rien.
- **Métrique** : MAE (erreur moyenne en unités) — à compléter si besoin.
- **Limite du backtest** : on ne peut pas observer les ventes à un prix jamais pratiqué. Sur données synthétiques, comparer à la vérité connue ; chez un vrai client, seul un **test contrôlé** (une partie des produits suit les recommandations, l'autre non) mesure le vrai gain de marge.

## Points de vigilance

- **Endogénéité des prix** : les prix historiques ne sont pas aléatoires ; signaler tout risque de confondre corrélation et causalité.
- **Peu de variations de prix** dans l'historique = élasticité peu fiable : le signaler au lieu de produire une recommandation trompeuse.
- **Pas d'extrapolation** : ne jamais recommander un prix en dehors de la plage de prix observée pour ce produit (ou le signaler explicitement comme incertain).
- Collecte des prix concurrents : ne rien coder avant d'avoir vérifié le cadre légal.

## Hypothèses business à tester (garder en tête)

- Les PME acceptent-elles qu'une IA recommande leurs prix ?
- Possèdent-elles suffisamment de données historiques ?
- Un gain de marge de 3–5 % est-il réellement observable ?
- Monitoring concurrent obligatoire ou pas ?
- Recommandations ou automatisation complète ?