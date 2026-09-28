"""Tests du générateur synthétique (src/data/synthetic.py)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.loader import prepare_sales_data
from src.data.synthetic import (
    BRAND_ELASTICITY_MULTIPLIERS,
    OUTPUT_COLUMNS,
    SyntheticConfig,
    black_friday,
    build_calendar,
    demand_mean,
    generate_synthetic_sales,
)


# Les jeux de données sont générés une seule fois par module (scope="module")
# pour que la suite de tests reste rapide.
@pytest.fixture(scope="module")
def endo():
    return generate_synthetic_sales(SyntheticConfig(endogenous=True, seed=1))


@pytest.fixture(scope="module")
def exo():
    return generate_synthetic_sales(SyntheticConfig(endogenous=False, seed=1))


# ---------------------------------------------------------------------------
# Formule de la demande (fonction pure, testable exactement)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("elasticity", [-0.8, -1.5, -2.5])
def test_doubler_le_prix_multiplie_la_demande_par_2_puissance_elasticite(elasticity):
    q1 = demand_mean(10, 50.0, 50.0, elasticity, competitor_price=50.0)
    q2 = demand_mean(10, 100.0, 50.0, elasticity, competitor_price=50.0)
    assert q2 / q1 == pytest.approx(2 ** elasticity)


def test_elasticite_locale_egale_au_parametre():
    """d log Q / d log p, mesuré numériquement, doit valoir l'élasticité choisie."""
    p, h = 40.0, 1e-4
    q_plus = demand_mean(10, p * np.exp(h), 50.0, -1.7, competitor_price=45.0, season=1.3)
    q_minus = demand_mean(10, p * np.exp(-h), 50.0, -1.7, competitor_price=45.0, season=1.3)
    assert (np.log(q_plus) - np.log(q_minus)) / (2 * h) == pytest.approx(-1.7)


def test_prix_de_reference_donne_la_demande_de_base():
    assert demand_mean(12.0, 50.0, 50.0, -2.0, competitor_price=50.0) == pytest.approx(12.0)


def test_promo_et_concurrent_font_monter_la_demande():
    base = demand_mean(10, 50.0, 50.0, -2.0, competitor_price=50.0)
    assert demand_mean(10, 50.0, 50.0, -2.0, competitor_price=50.0, is_promotion=1) > base
    assert demand_mean(10, 50.0, 50.0, -2.0, competitor_price=60.0) > base


# ---------------------------------------------------------------------------
# Calendrier
# ---------------------------------------------------------------------------

def test_dates_black_friday():
    # Mêmes dates que le dictionnaire du fichier fourni.
    assert black_friday(2024) == date(2024, 11, 29)
    assert black_friday(2025) == date(2025, 11, 28)


def test_calendrier_black_friday_et_noel():
    cal = build_calendar(pd.date_range("2024-11-27", "2024-12-26")).set_index("date")
    assert cal.loc["2024-11-29", "days_to_black_friday"] == 0
    assert cal.loc["2024-11-27", "days_to_black_friday"] == 2
    # Week-end du Black Friday : du vendredi 29/11 au lundi 02/12 inclus.
    assert cal.loc["2024-11-28":"2024-12-03", "is_black_friday"].tolist() == [0, 1, 1, 1, 1, 0]
    assert cal.loc["2024-12-24", "is_christmas_period"] == 1
    assert cal.loc["2024-12-25", "is_christmas_period"] == 0
    assert cal.loc["2024-12-25", "is_holiday"] == 1


# ---------------------------------------------------------------------------
# Format et cohérence des données générées
# ---------------------------------------------------------------------------

def test_meme_graine_memes_donnees():
    cfg = SyntheticConfig(products_per_category=1, n_days=60, seed=7)
    a = generate_synthetic_sales(cfg).sales
    b = generate_synthetic_sales(cfg).sales
    pd.testing.assert_frame_equal(a, b)


def test_graine_differente_donnees_differentes():
    a = generate_synthetic_sales(SyntheticConfig(products_per_category=1, n_days=60, seed=1)).sales
    b = generate_synthetic_sales(SyntheticConfig(products_per_category=1, n_days=60, seed=2)).sales
    assert not a["y_sales"].equals(b["y_sales"])


def test_meme_format_que_le_fichier_excel(endo):
    assert list(endo.sales.columns) == OUTPUT_COLUMNS
    cfg = SyntheticConfig()
    assert len(endo.sales) == 5 * cfg.products_per_category * cfg.n_days


def test_passe_tous_les_controles_du_chargeur(endo, exo):
    """Le chargeur refuse les fuites, doublons, ventes > stock, prix < coût…
    Si les données synthétiques passent, elles respectent toutes ces règles."""
    for data in (endo, exo):
        df = prepare_sales_data(data.sales)
        assert len(df) == len(data.sales)


def test_la_verite_n_est_pas_dans_les_donnees_observables(endo):
    """Anti-triche : le modèle ne doit jamais voir l'élasticité ni la vraie demande."""
    hidden = {"true_elasticity", "base_demand", "true_demand_mean", "reference_price", "conversion_rate"}
    assert hidden.isdisjoint(endo.sales.columns)
    assert {"true_elasticity", "base_demand"} <= set(endo.products.columns)


def test_elasticite_vraie_categorie_fois_marque(endo):
    cfg = SyntheticConfig()
    for _, prod in endo.products.iterrows():
        allowed = [cfg.category_elasticity[prod["category"]] * m for m in BRAND_ELASTICITY_MULTIPLIERS]
        assert np.isclose(prod["true_elasticity"], allowed).any()


def test_prix_jamais_sous_le_plancher_de_marge(endo, exo):
    for s in (endo.sales, exo.sales):
        floor = s["cost"] / (1 - s["target_margin"])
        assert (s["current_price"] >= floor - 1e-9).all()


def test_variation_de_prix_hors_promo_respecte_le_maximum(endo):
    """Entre deux jours consécutifs sans promo, le prix ne bouge pas plus que
    max_daily_price_change (tolérance d'un centime pour l'arrondi)."""
    s = endo.sales.sort_values(["product_id", "date"])
    prev_price = s.groupby("product_id")["current_price"].shift(1)
    prev_promo = s.groupby("product_id")["is_promotion"].shift(1)
    mask = (s["is_promotion"] == 0) & (prev_promo == 0)
    change = (s["current_price"] - prev_price).abs()[mask]
    allowed = (s["max_daily_price_change"] * prev_price)[mask] + 0.011
    assert (change <= allowed).all()


def test_ruptures_presentes_mais_rares(endo, exo):
    for data in (endo, exo):
        rate = (data.sales["y_sales"] >= data.sales["stock"]).mean()
        assert 0 < rate < 0.05


def test_le_prix_varie_assez_pour_mesurer_l_elasticite(endo):
    g = endo.sales.groupby("product_id")["current_price"]
    assert ((g.max() / g.min()) > 1.2).all()


# ---------------------------------------------------------------------------
# Endogénéité : le piège est-il bien présent quand on le demande ?
# ---------------------------------------------------------------------------

def _corr_price_vs_past_sales(sales: pd.DataFrame) -> float:
    """Corrélation, à l'intérieur de chaque produit et hors promo, entre le
    prix du jour et les ventes des 7 jours précédents."""
    lp = np.log(sales["current_price"])
    ls = np.log1p(sales["sales_7d"])
    lp = lp - lp.groupby(sales["product_id"]).transform("mean")
    ls = ls - ls.groupby(sales["product_id"]).transform("mean")
    regular = sales["is_promotion"] == 0
    return float(np.corrcoef(lp[regular], ls[regular])[0, 1])


def test_endogene_le_prix_suit_la_demande_passee(endo, exo):
    # Exogène : prix haut → moins de ventes → corrélation NÉGATIVE (effet causal normal).
    # Endogène : le marchand monte le prix quand ça se vend → corrélation POSITIVE.
    assert _corr_price_vs_past_sales(exo.sales) < 0
    assert _corr_price_vs_past_sales(endo.sales) > 0


def test_endogene_promos_concentrees_sur_le_black_friday(endo, exo):
    bf_endo = endo.sales.loc[endo.sales["is_black_friday"] == 1, "is_promotion"].mean()
    bf_exo = exo.sales.loc[exo.sales["is_black_friday"] == 1, "is_promotion"].mean()
    assert bf_endo > 0.6
    assert bf_exo < 0.3
