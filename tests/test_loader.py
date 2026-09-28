"""Tests du module de chargement (src/data/loader.py).

Deux familles de tests :
- des tests « unitaires » sur de petits DataFrames construits à la main :
  on connaît exactement la bonne réponse, donc on peut vérifier chaque règle ;
- des tests « d'intégration » sur le vrai fichier Excel fictif : on vérifie
  que la chaîne complète fonctionne sur des données réalistes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    GUARDRAIL_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET,
    add_stockout_flag,
    load_products_excel,
    load_sales_excel,
    prepare_sales_data,
    price_variation_summary,
    validate_sales_data,
)

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "dataset_ventes_fictif_v2.xlsx"


# ---------------------------------------------------------------------------
# Petit jeu de données fabriqué à la main
# ---------------------------------------------------------------------------

def make_raw(n_days: int = 10, sales=None) -> pd.DataFrame:
    """Construit un historique cohérent d'UN produit sur ``n_days`` jours.

    ``sales_7d`` est calculé correctement (J-7 à J-1). Pour les 7 premiers
    jours, l'historique est incomplet ; on met une valeur quelconque, comme
    dans un vrai export (le contrôle ignore ces lignes).
    """
    if sales is None:
        sales = [3, 5, 2, 4, 6, 1, 0, 7, 2, 5][:n_days]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    s = pd.Series(sales, dtype=float)
    sales_7d = s.shift(1).rolling(7).sum().fillna(20).astype(int)
    return pd.DataFrame({
        "date": dates,
        "product_id": "P001",
        "category": "Accessoires",
        "brand": "Kuvo",
        "cost": 10.0,
        "rating": 4.5,
        "current_price": 20.0,
        "competitor_min_price": 21.0,
        "stock": 50,
        "stock_coverage": 15.0,
        "is_promotion": 0,
        "day_of_week": dates.day_name(),
        "month": dates.month,
        "is_black_friday": 0,
        "days_to_black_friday": 300,
        "is_christmas_period": 0,
        "is_holiday": 0,
        "views_7d": 500,
        "sales_7d": sales_7d,
        "y_sales": sales,  # nom utilisé dans le fichier Excel
        "target_margin": 0.3,
        "max_daily_price_change": 0.05,
        "repricing_frequency": 1,
    })


# ---------------------------------------------------------------------------
# Tests unitaires
# ---------------------------------------------------------------------------

def test_prepare_renomme_la_cible_et_ajoute_is_stockout():
    df = prepare_sales_data(make_raw())
    assert TARGET in df.columns
    assert "y_sales" not in df.columns
    assert "is_stockout" in df.columns


def test_prepare_ne_modifie_pas_le_dataframe_recu():
    raw = make_raw()
    copie = raw.copy()
    prepare_sales_data(raw)
    pd.testing.assert_frame_equal(raw, copie)


def test_prepare_trie_par_produit_puis_date():
    raw = make_raw().sample(frac=1, random_state=0)  # on mélange les lignes
    df = prepare_sales_data(raw)
    assert df["date"].is_monotonic_increasing


def test_day_of_week_est_une_categorie_ordonnee():
    df = prepare_sales_data(make_raw())
    assert isinstance(df["day_of_week"].dtype, pd.CategoricalDtype)
    assert df["day_of_week"].cat.categories[0] == "Monday"


def test_stockout_quand_les_ventes_atteignent_le_stock():
    df = pd.DataFrame({TARGET: [3, 5, 0, 0], "stock": [10, 5, 0, 4]})
    # ligne 1 : tout le stock vendu → rupture ; ligne 2 : stock nul → rupture
    assert add_stockout_flag(df)["is_stockout"].tolist() == [0, 1, 1, 0]


def test_donnees_propres_aucun_probleme():
    df = make_raw().rename(columns={"y_sales": TARGET})
    assert validate_sales_data(df) == []


def test_detecte_colonne_manquante():
    df = make_raw().rename(columns={"y_sales": TARGET}).drop(columns=["cost"])
    problems = validate_sales_data(df)
    assert len(problems) == 1 and "cost" in problems[0]


def test_detecte_doublons():
    df = make_raw().rename(columns={"y_sales": TARGET})
    df = pd.concat([df, df.iloc[[0]]])
    assert any("doublon" in p for p in validate_sales_data(df))


def test_detecte_prix_sous_le_cout():
    df = make_raw().rename(columns={"y_sales": TARGET})
    df.loc[3, "current_price"] = 5.0  # coût = 10
    assert any("sous le coût" in p for p in validate_sales_data(df))


def test_detecte_ventes_superieures_au_stock():
    df = make_raw().rename(columns={"y_sales": TARGET})
    df.loc[2, "stock"] = 1  # ventes = 2
    assert any("supérieures au stock" in p for p in validate_sales_data(df))


def test_detecte_fuite_dans_sales_7d():
    """Si sales_7d inclut les ventes du jour J, c'est une fuite : on doit la voir."""
    df = make_raw().rename(columns={"y_sales": TARGET})
    s = df[TARGET]
    df["sales_7d"] = s.rolling(7).sum().fillna(20).astype(int)  # J-6..J → fuite !
    assert any("sales_7d" in p for p in validate_sales_data(df))


def test_prepare_refuse_des_donnees_invalides():
    raw = make_raw()
    raw.loc[0, "current_price"] = -1.0
    with pytest.raises(ValueError, match="Données refusées"):
        prepare_sales_data(raw)


def test_price_variation_summary():
    df = pd.DataFrame({
        "product_id": ["A", "A", "B", "B"],
        "current_price": [10.0, 20.0, 5.0, 5.0],
    })
    summary = price_variation_summary(df)
    assert summary.loc["A", "price_range_ratio"] == 2.0
    assert summary.loc["B", "price_cv"] == 0.0  # prix jamais modifié
    assert summary.loc["B", "n_distinct_prices"] == 1


# ---------------------------------------------------------------------------
# Tests d'intégration sur le dataset fictif fourni
# ---------------------------------------------------------------------------

needs_dataset = pytest.mark.skipif(not DATASET_PATH.exists(), reason="dataset fictif absent")


@pytest.fixture(scope="module")
def sales():
    # scope="module" : le fichier (5 Mo) n'est lu qu'une fois pour tous les tests.
    return load_sales_excel(DATASET_PATH)


@needs_dataset
def test_dataset_forme_attendue(sales):
    assert sales.shape[0] == 36_200
    assert sales["product_id"].nunique() == 50
    assert set(REQUIRED_COLUMNS) <= set(sales.columns)
    assert set(GUARDRAIL_COLUMNS) <= set(sales.columns)


@needs_dataset
def test_dataset_ruptures_rares(sales):
    # Option A (exclure les ruptures) n'est raisonnable que si elles sont rares.
    assert 0 < sales["is_stockout"].mean() < 0.05


@needs_dataset
def test_dataset_prix_ont_varie(sales):
    summary = price_variation_summary(sales)
    assert (summary["n_distinct_prices"] > 1).all()


@needs_dataset
def test_onglet_produits():
    products = load_products_excel(DATASET_PATH)
    assert len(products) == 50  # les lignes de notes ont été retirées
    assert {"base_demand", "price_sensitivity", "reference_price"} <= set(products.columns)
    assert set(products["price_sensitivity"]) == {"Très forte", "Forte", "Moyenne", "Faible"}
