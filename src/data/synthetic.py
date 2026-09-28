"""Générateur de données de ventes synthétiques, à élasticité-prix CONNUE.

Pourquoi ce module ? Avec de vraies données (ou le dataset fictif fourni), on
ne connaît pas la « bonne réponse » : on ne peut pas vérifier que le modèle a
compris l'effet du prix. Ici, c'est nous qui fabriquons les ventes à partir
d'une formule dont on choisit l'élasticité. C'est l'« examen corrigé » du
modèle de demande (étape 2).

Le générateur produit un tableau AU MÊME FORMAT que l'onglet « Dataset » du
fichier Excel (même colonnes, cible ``y_sales``), pour que tout le reste du
code (chargement, contrôles, modèle) fonctionne sans cas particulier.

Formule de la demande moyenne (produit i, jour t)
-------------------------------------------------
    λ = demande_de_base
        × (prix / prix_référence) ^ élasticité        ← effet du prix (ce qu'on veut retrouver)
        × (prix_concurrent / prix_référence) ^ 0.8     ← effet du concurrent
        × saison(jour de semaine, fériés, Black Friday, Noël)
        × (1 + bonus_promo si promotion)
        × (note / 4.2)
        × exp(tendance)                                ← choc de demande persistant, non observé

Les ventes sont ensuite tirées au hasard autour de λ, puis plafonnées par le
stock disponible.

Pourquoi l'effet du concurrent ne dépend-il pas de NOTRE prix ? Pour que
l'élasticité « propre » (à prix concurrent fixé) soit EXACTEMENT le chiffre
choisi : d log λ / d log prix = élasticité. C'est ce qu'on vérifiera.

Endogénéité (option ``endogenous=True``)
----------------------------------------
Dans la réalité, le marchand ne fixe pas ses prix au hasard : il les monte
quand ça se vend bien et fait des promos quand il a trop de stock ou pendant
le Black Friday. Le prix est alors corrélé à des chocs de demande que le
modèle ne voit pas directement → un modèle naïf confond corrélation et
causalité. On fabrique volontairement ce piège pour vérifier que le modèle
n'y tombe pas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import NamedTuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------

# Élasticité par catégorie : même ordre que l'onglet « Produits » du fichier
# fourni (Accessoires très sensible … Luxe peu sensible), mais cette fois en
# chiffres. Exemple : −2.0 signifie « +1 % de prix → −2 % de ventes ».
DEFAULT_CATEGORY_ELASTICITY = {
    "Accessoires": -2.5,
    "Sport": -2.0,
    "Électronique": -1.5,
    "Maison": -1.5,
    "Luxe": -0.8,
}

# 3 marques par catégorie : premium / standard / économique. La marque module
# l'élasticité (les clients d'une marque premium sont moins sensibles au prix),
# comme dans le dictionnaire du fichier fourni.
DEFAULT_BRANDS = {
    "Accessoires": ["Kuvo", "Linko", "Strapz"],
    "Sport": ["Stridex", "Movo", "ActiFit"],
    "Électronique": ["VoltaTech", "Nexio", "Ampira"],
    "Maison": ["Lumea", "Casanova", "HomeLine"],
    "Luxe": ["Maison Arlo", "Aurelis", "Velmont"],
}
BRAND_ELASTICITY_MULTIPLIERS = (0.8, 1.0, 1.2)

# Ordre de grandeur du prix de référence par catégorie (€, bornes min / max).
_PRICE_RANGE = {
    "Accessoires": (8, 60),
    "Sport": (20, 320),
    "Électronique": (50, 550),
    "Maison": (20, 170),
    "Luxe": (230, 1500),
}

# Jours fériés belges à date fixe (mois, jour). Les fériés mobiles (Pâques,
# Ascension, Pentecôte) sont ignorés : c'est une simplification assumée.
_BELGIAN_FIXED_HOLIDAYS = [(1, 1), (5, 1), (7, 21), (8, 15), (11, 1), (11, 11), (12, 25)]

# Colonnes produites, dans le même ordre que l'onglet « Dataset » de l'Excel.
OUTPUT_COLUMNS = [
    "date", "product_id", "category", "brand", "cost", "rating",
    "current_price", "competitor_min_price", "stock", "stock_coverage",
    "is_promotion", "day_of_week", "month", "is_black_friday",
    "days_to_black_friday", "is_christmas_period", "is_holiday",
    "views_7d", "sales_7d", "target_margin", "max_daily_price_change",
    "repricing_frequency", "y_sales",
]


@dataclass
class SyntheticConfig:
    """Tous les réglages du générateur, rassemblés au même endroit.

    Un ``dataclass`` est une classe qui sert juste à ranger des paramètres
    avec des valeurs par défaut. On peut en changer un seul :
    ``SyntheticConfig(endogenous=False)``.
    """

    products_per_category: int = 4
    start_date: str = "2024-01-01"
    n_days: int = 730                     # 2 ans → deux Black Friday, deux Noëls
    warmup_days: int = 7                  # jours simulés puis retirés (sales_7d incomplet)
    category_elasticity: dict = field(default_factory=lambda: dict(DEFAULT_CATEGORY_ELASTICITY))
    cross_elasticity: float = 0.8         # effet du prix concurrent
    promo_uplift: float = 0.20            # +20 % de demande en promo, EN PLUS de la baisse de prix
    endogenous: bool = True               # prix qui réagissent à la demande (voir docstring du module)
    dispersion: float = 10.0              # plus c'est petit, plus les ventes sont bruitées
    trend_persistence: float = 0.9        # mémoire du choc de demande (0 = aucun, 1 = permanent)
    trend_volatility: float = 0.08
    seed: int = 42


class SyntheticData(NamedTuple):
    """Résultat du générateur.

    - ``sales`` : les données « observables », au format de l'Excel. C'est
      tout ce que le modèle aura le droit de voir.
    - ``products`` : la fiche de chaque produit, AVEC la vérité cachée
      (``true_elasticity``, ``base_demand``).
    - ``truth`` : la demande moyenne réelle λ de chaque jour, avant hasard et
      avant plafonnement par le stock. Sert au backtest (étape 5).
    """

    sales: pd.DataFrame
    products: pd.DataFrame
    truth: pd.DataFrame


# ---------------------------------------------------------------------------
# Calendrier
# ---------------------------------------------------------------------------

def black_friday(year: int) -> date:
    """Black Friday = lendemain du 4e jeudi de novembre (Thanksgiving).

    Attention : ce n'est PAS toujours le 4e vendredi (ex. 2024 : le 1er
    novembre est un vendredi, le 4e vendredi est le 22, le Black Friday le 29).
    """
    first_nov = date(year, 11, 1)
    # weekday() : lundi = 0 … jeudi = 3
    first_thursday = first_nov + timedelta(days=(3 - first_nov.weekday()) % 7)
    return first_thursday + timedelta(weeks=3, days=1)


def build_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Calcule les colonnes de calendrier pour une liste de dates.

    Ces colonnes ne dépendent que de la date : on les calcule une fois, puis
    on les partage entre tous les produits.
    """
    cal = pd.DataFrame({"date": dates})
    cal["day_of_week"] = dates.day_name()
    cal["month"] = dates.month

    is_bf, days_to_bf = [], []
    for d in dates.date:
        bf = black_friday(d.year)
        # Week-end du Black Friday : du vendredi au lundi (Cyber Monday).
        is_bf.append(int(bf <= d <= bf + timedelta(days=3)))
        # Jours jusqu'au PROCHAIN Black Friday (0 le jour même).
        nxt = bf if d <= bf else black_friday(d.year + 1)
        days_to_bf.append((nxt - d).days)
    cal["is_black_friday"] = is_bf
    cal["days_to_black_friday"] = days_to_bf

    cal["is_christmas_period"] = ((dates.month == 12) & (dates.day <= 24)).astype(int)
    holidays = set(_BELGIAN_FIXED_HOLIDAYS)
    cal["is_holiday"] = [int((d.month, d.day) in holidays) for d in dates]
    return cal


def seasonal_multiplier(cal: pd.DataFrame, category: str) -> np.ndarray:
    """Effet de la saison sur la demande (1.0 = jour normal).

    Les valeurs sont choisies pour ressembler au dictionnaire du fichier
    fourni : creux avant le Black Friday (les clients attendent), pic pendant,
    montée à Noël (forte sur le Luxe), fériés +25 % sauf 25/12 et 01/01.
    """
    m = np.ones(len(cal))
    m *= np.where(cal["day_of_week"].isin(["Saturday", "Sunday"]), 1.10, 1.0)
    m *= np.where(cal["is_black_friday"] == 1, 2.0, 1.0)
    pre_bf = (cal["days_to_black_friday"] >= 1) & (cal["days_to_black_friday"] <= 7)
    m *= np.where(pre_bf, 0.85, 1.0)
    christmas_boost = 1.8 if category == "Luxe" else 1.3
    m *= np.where(cal["is_christmas_period"] == 1, christmas_boost, 1.0)
    dead_days = cal["date"].dt.strftime("%m-%d").isin(["12-25", "01-01"])
    m *= np.where(cal["is_holiday"] == 1, np.where(dead_days, 0.4, 1.25), 1.0)
    return m


# ---------------------------------------------------------------------------
# Formule de la demande
# ---------------------------------------------------------------------------

def demand_mean(
    base_demand: float,
    price,
    reference_price: float,
    elasticity: float,
    competitor_price,
    season=1.0,
    is_promotion=0,
    rating: float = 4.2,
    trend=0.0,
    cross_elasticity: float = 0.8,
    promo_uplift: float = 0.20,
):
    """Demande moyenne λ (ventes attendues par jour). Voir la formule en tête de module.

    Fonction « pure » (aucun hasard) : on peut la tester exactement, et le
    backtest pourra l'appeler pour connaître la vraie demande à n'importe quel
    prix.
    """
    price = np.asarray(price, dtype=float)
    competitor_price = np.asarray(competitor_price, dtype=float)
    return (
        base_demand
        * (price / reference_price) ** elasticity
        * (competitor_price / reference_price) ** cross_elasticity
        * season
        * (1 + promo_uplift * np.asarray(is_promotion))
        * (rating / 4.2)
        * np.exp(trend)
    )


# ---------------------------------------------------------------------------
# Catalogue de produits
# ---------------------------------------------------------------------------

def make_products(config: SyntheticConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Tire au hasard les caractéristiques de chaque produit (vérité incluse)."""
    rows = []
    k = 1
    for category, cat_elasticity in config.category_elasticity.items():
        brands = DEFAULT_BRANDS.get(category, [f"{category}-{j}" for j in range(3)])
        lo, hi = _PRICE_RANGE.get(category, (20, 200))
        for j in range(config.products_per_category):
            brand_idx = j % 3
            reference_price = round(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), 2)
            # Coût = 35 à 60 % du prix de référence, marge cible ≤ 25 % :
            # le plancher de marge reste ≤ 80 % du prix de référence, ce qui
            # laisse de la place pour faire varier le prix.
            cost = round(reference_price * rng.uniform(0.35, 0.60), 2)
            rows.append({
                "product_id": f"S{k:03d}",
                "category": category,
                "brand": brands[brand_idx],
                "true_elasticity": cat_elasticity * BRAND_ELASTICITY_MULTIPLIERS[brand_idx],
                "base_demand": round(float(rng.uniform(3, 30)), 1),
                "reference_price": reference_price,
                "cost": cost,
                "rating": round(float(rng.uniform(3.5, 4.9)), 1),
                "target_margin": float(rng.choice([0.05, 0.10, 0.15, 0.20, 0.25])),
                "max_daily_price_change": float(rng.choice([0.03, 0.05, 0.08, 0.10])),
                "repricing_frequency": int(rng.choice([1, 2, 7])),
                "conversion_rate": float(rng.uniform(0.02, 0.06)),  # ventes / vues
            })
            k += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Simulation jour par jour
# ---------------------------------------------------------------------------

def _simulate_product(
    prod: pd.Series,
    cal: pd.DataFrame,
    season: np.ndarray,
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simule l'historique complet d'un produit.

    On avance jour par jour (boucle) car chaque jour dépend des précédents :
    le stock restant, la tendance, le prix d'hier, les ventes des 7 derniers
    jours (qui influencent le prix en mode endogène).
    """
    prod = prod.to_dict()   # dictionnaire : accès plus rapide qu'une Series dans la boucle
    n = len(cal)
    p_ref = prod["reference_price"]
    floor = prod["cost"] / (1 - prod["target_margin"])   # prix minimum autorisé
    ceiling = 1.35 * p_ref
    base = prod["base_demand"]

    # Tableaux de sortie (remplis au fur et à mesure).
    price = np.empty(n)
    competitor = np.empty(n)
    promo = np.zeros(n, dtype=int)
    stock = np.empty(n, dtype=int)
    sales = np.empty(n, dtype=int)
    views = np.empty(n, dtype=int)
    lam = np.empty(n)

    # États initiaux.
    regular_price = p_ref * rng.uniform(0.95, 1.05)       # prix « hors promo »
    comp_level = p_ref * rng.uniform(0.95, 1.05)
    trend = 0.0
    current_stock = int(30 * base)
    pending_orders: list[tuple[int, int]] = []            # (jour d'arrivée, quantité)
    promo_days_left, promo_discount = 0, 0.0
    # Tableau numpy plutôt que colonne pandas : beaucoup plus rapide dans une boucle.
    is_bf = cal["is_black_friday"].to_numpy()

    for t in range(n):
        # --- 1. Réapprovisionnement : les commandes arrivées ce matin. ---
        arrived = sum(q for day, q in pending_orders if day == t)
        current_stock += arrived
        pending_orders = [(day, q) for day, q in pending_orders if day > t]

        # --- 2. Indicateurs « passés » connus du marchand ce matin. ---
        past_sales_7 = sales[max(0, t - 7):t].sum() if t > 0 else 7 * base
        daily_rate = max(past_sales_7 / 7, 0.1)
        coverage = current_stock / daily_rate

        # --- 3. Prix concurrent : marche aléatoire, −10 à −20 % au Black Friday. ---
        comp_level *= np.exp(rng.normal(0, 0.01) - 0.02 * np.log(comp_level / p_ref))
        competitor[t] = comp_level * (1 - rng.uniform(0.10, 0.20) if is_bf[t] else 1)

        # --- 4. Repricing (seulement les jours autorisés, hors promo). ---
        if t % prod["repricing_frequency"] == 0 and promo_days_left == 0:
            change = rng.normal(0, prod["max_daily_price_change"] / 2)   # part aléatoire
            change -= 0.1 * np.log(regular_price / p_ref)                  # retour vers p_ref
            if config.endogenous:
                # Le marchand monte le prix quand ça se vend bien, et selon son stock.
                change += 0.3 * np.log(daily_rate / base)
                if coverage < 10:
                    change += prod["max_daily_price_change"] / 2
                elif coverage > 40:
                    change -= prod["max_daily_price_change"] / 2
            change = np.clip(change, -prod["max_daily_price_change"], prod["max_daily_price_change"])
            regular_price = float(np.clip(regular_price * (1 + change), floor, ceiling))

        # --- 5. Promotions. ---
        if promo_days_left == 0:
            if config.endogenous:
                if is_bf[t]:
                    p_start = 0.7        # promo quasi systématique au Black Friday
                elif coverage > 40:
                    p_start = 0.15       # trop de stock → on déstocke
                elif coverage < 10:
                    p_start = 0.0        # stock bas → pas de promo
                else:
                    p_start = 0.01
            else:
                p_start = 0.02           # promos au hasard, sans lien avec la demande
            if rng.random() < p_start:
                promo_days_left = int(rng.integers(3, 8))
                promo_discount = rng.uniform(0.10, 0.30)
        if promo_days_left > 0:
            promo[t] = 1
            promo_days_left -= 1
        # Arrondi au centime, sans jamais passer sous le plancher de marge.
        raw_price = regular_price * (1 - promo_discount) if promo[t] else regular_price
        price[t] = max(round(raw_price, 2), np.ceil(floor * 100) / 100)

        # --- 6. Demande et ventes. ---
        trend = config.trend_persistence * trend + rng.normal(0, config.trend_volatility)
        lam[t] = demand_mean(
            base, price[t], p_ref, prod["true_elasticity"], competitor[t],
            season=season[t], is_promotion=promo[t], rating=prod["rating"], trend=trend,
            cross_elasticity=config.cross_elasticity, promo_uplift=config.promo_uplift,
        )
        # Loi « Poisson surdispersée » (Gamma-Poisson = binomiale négative) :
        # les ventes réelles varient plus qu'une simple loi de Poisson.
        demand = rng.poisson(lam[t] * rng.gamma(config.dispersion, 1 / config.dispersion))
        stock[t] = current_stock
        sales[t] = min(demand, current_stock)     # on ne vend pas ce qu'on n'a pas
        current_stock -= sales[t]

        # Vues de la fiche produit : proportionnelles à l'intérêt (hors effet prix).
        interest = lam[t] / (price[t] / p_ref) ** prod["true_elasticity"]
        views[t] = rng.poisson(interest / prod["conversion_rate"])

        # --- 7. Commande fournisseur si le stock passe sous ~12 jours de vente. ---
        if current_stock < 12 * base and not pending_orders:
            lead_time = int(rng.integers(3, 11))  # délai parfois long → ruptures
            pending_orders.append((t + 1 + lead_time, int(30 * base)))

    out = cal.copy()
    out["product_id"] = prod["product_id"]
    out["current_price"] = price
    out["competitor_min_price"] = np.round(competitor, 2)
    out["is_promotion"] = promo
    out["stock"] = stock
    out["y_sales"] = sales
    out["true_demand_mean"] = lam
    # Features « 7 derniers jours » : shift(1) → le jour J n'est JAMAIS inclus.
    out["sales_7d"] = pd.Series(sales).shift(1).rolling(7).sum().to_numpy()
    out["views_7d"] = pd.Series(views).shift(1).rolling(7).sum().to_numpy()
    return out


def generate_synthetic_sales(config: SyntheticConfig | None = None) -> SyntheticData:
    """Point d'entrée : génère le catalogue, puis l'historique de chaque produit."""
    config = config or SyntheticConfig()
    # Un seul générateur aléatoire, initialisé avec la graine : même graine
    # → exactement les mêmes données (tests reproductibles).
    rng = np.random.default_rng(config.seed)

    products = make_products(config, rng)
    dates = pd.date_range(config.start_date, periods=config.warmup_days + config.n_days, freq="D")
    cal = build_calendar(dates)

    frames = []
    for _, prod in products.iterrows():
        season = seasonal_multiplier(cal, prod["category"])
        frames.append(_simulate_product(prod, cal, season, config, rng))
    df = pd.concat(frames, ignore_index=True)

    # On retire les jours de « chauffe » : leur sales_7d est incomplet.
    df = df[df["date"] >= dates[config.warmup_days]].reset_index(drop=True)
    df["sales_7d"] = df["sales_7d"].astype(int)
    df["views_7d"] = df["views_7d"].astype(int)

    # stock_coverage = jours de vente couverts au rythme des 7 derniers jours
    # (formule du dictionnaire), plafonné à 365.
    rate = df["sales_7d"] / 7
    df["stock_coverage"] = np.where(rate > 0, np.minimum(df["stock"] / rate.where(rate > 0), 365), 365.0)

    # Colonnes constantes par produit (fiche produit).
    static = ["category", "brand", "cost", "rating", "target_margin",
              "max_daily_price_change", "repricing_frequency"]
    df = df.merge(products[["product_id", *static]], on="product_id")

    truth = df[["date", "product_id", "true_demand_mean"]].copy()
    sales = df[OUTPUT_COLUMNS].copy()   # uniquement les colonnes observables
    return SyntheticData(sales=sales, products=products, truth=truth)


if __name__ == "__main__":
    # Usage : python -m src.data.synthetic  → écrit un CSV dans data/synthetic/
    # (dossier ignoré par Git : on régénère plutôt que de versionner).
    from pathlib import Path

    out_dir = Path(__file__).resolve().parents[2] / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = generate_synthetic_sales()
    data.sales.to_csv(out_dir / "sales.csv", index=False)
    data.products.to_csv(out_dir / "products.csv", index=False)
    print(f"{len(data.sales)} lignes écrites dans {out_dir}")
