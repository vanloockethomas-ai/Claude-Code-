"""Chargement et contrôle qualité des données de ventes.

Rôle de ce module : transformer un fichier brut (ici l'Excel fictif) en un
DataFrame propre, au format attendu par le reste du projet (voir CLAUDE.md),
et REFUSER les données qui présentent un problème grave (fuite de la cible,
doublons, prix sous le plancher de marge…).

Pourquoi refuser plutôt que corriger en silence ? Parce qu'un modèle entraîné
sur des données fausses donne des recommandations fausses, sans prévenir.
Mieux vaut une erreur claire au chargement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rôle de chaque colonne
# ---------------------------------------------------------------------------
# On range les colonnes par rôle une fois pour toutes. Le modèle de demande,
# l'optimiseur et les tests s'appuieront sur ces listes : si une colonne change
# de rôle, on ne la modifie qu'ici.

# Identifiants : servent à trier / découper, JAMAIS à prédire.
ID_COLUMNS = ["date", "product_id"]

# Cible : ce que le modèle doit prédire.
TARGET = "units_sold"

# Garde-fous commerciaux : utilisés par l'optimiseur, pas par le modèle.
# (Ils sont constants par produit : les donner au modèle reviendrait à lui
# donner un identifiant produit déguisé.)
GUARDRAIL_COLUMNS = ["target_margin", "max_daily_price_change", "repricing_frequency"]

# Colonnes minimales qu'un fichier doit contenir pour être accepté.
REQUIRED_COLUMNS = [
    "date", "product_id", "category", "brand", "cost", "rating",
    "current_price", "competitor_min_price", "stock", "stock_coverage",
    "is_promotion", "day_of_week", "month", "is_black_friday",
    "days_to_black_friday", "is_christmas_period", "is_holiday",
    "views_7d", "sales_7d", TARGET,
]

# Nom de la cible dans le fichier Excel fourni → nom standard du projet.
_RENAME_EXCEL = {"y_sales": TARGET}

# Colonnes de l'onglet « Produits » → noms courts en snake_case.
_RENAME_PRODUCTS = {
    "Sensibilité au prix": "price_sensitivity",
    "Demande de base (ventes/jour)": "base_demand",
    "Ventes moyennes observées": "observed_mean_sales",
    "Prix de référence": "reference_price",
}

# Ordre des jours, pour que « day_of_week » soit une catégorie ordonnée
# (Monday < Tuesday < …) plutôt qu'un texte trié par ordre alphabétique.
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

def add_stockout_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne ``is_stockout`` (0/1).

    Dans nos données, ``stock`` est le stock en DÉBUT de journée. Si on a
    vendu tout ce stock (``units_sold >= stock``), le produit s'est retrouvé
    en rupture : la vraie demande était peut-être plus élevée, mais on ne
    l'observe pas (on parle de demande « censurée »).

    Ces jours seront exclus de l'entraînement (option A validée pour la v1).
    Ici, on se contente de les REPÉRER.
    """
    out = df.copy()  # on ne modifie jamais le DataFrame reçu (évite les surprises)
    out["is_stockout"] = (out[TARGET] >= out["stock"]).astype(int)
    return out


def _expected_sales_7d(df: pd.DataFrame) -> pd.Series:
    """Recalcule ``sales_7d`` « sans fuite » : somme des ventes de J-7 à J-1.

    - ``groupby("product_id")`` : chaque produit a son propre historique.
    - ``shift(1)`` : décale d'un jour → la ligne J voit la vente de J-1,
      jamais celle de J. C'est ce décalage qui empêche la fuite.
    - ``rolling(7).sum()`` : somme glissante sur 7 jours.

    Les 7 premiers jours d'un produit n'ont pas d'historique complet : le
    résultat y vaut NaN et ces lignes ne sont pas contrôlées.
    """
    # On travaille sur une copie réindexée 0..n-1 : l'index d'origine peut
    # contenir des doublons (justement ce qu'on cherche aussi à détecter).
    work = df.reset_index(drop=True)
    ordered = work.sort_values(["product_id", "date"])
    expected = ordered.groupby("product_id")[TARGET].transform(
        lambda s: s.shift(1).rolling(7).sum()
    )
    # On renvoie les valeurs dans l'ordre des lignes de ``df``.
    return pd.Series(expected.sort_index().to_numpy(), index=df.index)


# ---------------------------------------------------------------------------
# Contrôle qualité
# ---------------------------------------------------------------------------

def validate_sales_data(df: pd.DataFrame) -> list[str]:
    """Vérifie les données et renvoie la liste des problèmes trouvés.

    Une liste vide signifie « tout va bien ». On renvoie une liste (plutôt que
    de s'arrêter au premier problème) pour que l'utilisateur voie TOUS les
    problèmes d'un coup.
    """
    problems: list[str] = []

    # 1. Colonnes manquantes : inutile d'aller plus loin sans elles.
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"Colonnes manquantes : {missing}"]

    # 2. Valeurs manquantes.
    n_nan = int(df[REQUIRED_COLUMNS].isna().sum().sum())
    if n_nan:
        problems.append(f"{n_nan} valeur(s) manquante(s) dans les colonnes obligatoires")

    # 3. Doublons : un produit ne doit avoir qu'une ligne par jour.
    n_dup = int(df.duplicated(["product_id", "date"]).sum())
    if n_dup:
        problems.append(f"{n_dup} doublon(s) (product_id, date)")

    # 4. Valeurs impossibles.
    if (df["current_price"] <= 0).any():
        problems.append("Prix nul ou négatif détecté")
    if (df[TARGET] < 0).any():
        problems.append("Ventes négatives détectées")
    if (df[TARGET] > df["stock"]).any():
        problems.append("Ventes supérieures au stock disponible")

    # 5. Jamais de prix sous le coût (contrainte stricte de CLAUDE.md).
    n_below_cost = int((df["current_price"] < df["cost"]).sum())
    if n_below_cost:
        problems.append(f"{n_below_cost} ligne(s) avec un prix sous le coût")

    # 6. Fuite de la cible dans sales_7d : on recalcule la valeur attendue
    #    (jours passés uniquement) et on compare.
    expected = _expected_sales_7d(df)
    checkable = expected.notna().to_numpy()
    n_leak = int((df["sales_7d"].to_numpy()[checkable] != expected.to_numpy()[checkable]).sum())
    if n_leak:
        problems.append(
            f"sales_7d incohérent sur {n_leak} ligne(s) : il doit valoir la somme "
            "des ventes de J-7 à J-1 (sinon risque de fuite de la cible)"
        )

    return problems


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def prepare_sales_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Met un DataFrame brut au format standard du projet, puis le contrôle.

    Séparé de la lecture du fichier pour pouvoir être testé sur de petits
    DataFrames construits à la main (sans fichier Excel).
    """
    df = raw.rename(columns=_RENAME_EXCEL).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["product_id"] = df["product_id"].astype(str)

    # Tri stable par produit puis par date : indispensable pour tous les
    # calculs « jours passés » (fenêtres glissantes, découpage temporel).
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    problems = validate_sales_data(df)
    if problems:
        raise ValueError("Données refusées :\n- " + "\n- ".join(problems))

    df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=_DAYS, ordered=True)
    df = add_stockout_flag(df)
    return df


def load_sales_excel(path: str | Path, sheet_name: str = "Dataset") -> pd.DataFrame:
    """Lit l'onglet des ventes d'un fichier Excel et le prépare."""
    raw = pd.read_excel(path, sheet_name=sheet_name)
    return prepare_sales_data(raw)


def load_products_excel(path: str | Path, sheet_name: str = "Produits") -> pd.DataFrame:
    """Lit l'onglet « Produits » (informations de référence, une ligne par produit).

    ATTENTION : ``base_demand`` et ``price_sensitivity`` décrivent comment les
    données fictives ont été fabriquées. Ce ne sont PAS des features : elles
    servent uniquement à vérifier ce que le modèle apprend.
    """
    products = pd.read_excel(path, sheet_name=sheet_name).rename(columns=_RENAME_PRODUCTS)
    # L'onglet contient des lignes de notes sous le tableau : on ne garde que
    # les lignes dont l'identifiant ressemble à « P001 ».
    is_product = products["product_id"].astype(str).str.fullmatch(r"P\d+")
    products = products[is_product].reset_index(drop=True)
    products["repricing_frequency"] = products["repricing_frequency"].astype(int)
    return products


def price_variation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Résume, par produit, à quel point le prix a varié dans l'historique.

    Utile pour le point de vigilance de CLAUDE.md : si un prix n'a presque
    jamais bougé, on ne peut pas mesurer de façon fiable l'effet du prix.

    - ``price_cv`` : coefficient de variation = écart-type / moyenne.
    - ``price_min`` / ``price_max`` : plage observée (l'optimiseur ne devra
      pas en sortir, pour ne pas extrapoler).
    """
    g = df.groupby("product_id")["current_price"]
    summary = pd.DataFrame({
        "price_min": g.min(),
        "price_max": g.max(),
        "price_cv": g.std() / g.mean(),
        "n_distinct_prices": g.nunique(),
    })
    summary["price_range_ratio"] = summary["price_max"] / summary["price_min"]
    return summary.replace([np.inf, -np.inf], np.nan)
