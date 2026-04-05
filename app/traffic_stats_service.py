import pandas as pd

from app.database import get_engine


def _load_traffic_data() -> pd.DataFrame:
    query = """
    SELECT
        id_transaction,
        date_heure,
        gare,
        classe_vehicule,
        type_paiement,
        statut_abonnement,
        montant_net
    FROM fact_transactions
    WHERE montant_net > 0
    """
    df = pd.read_sql(query, get_engine())
    df["date_heure"] = pd.to_datetime(df["date_heure"])
    return df


def _series_to_dict(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.items()}


def _series_to_float_dict(series: pd.Series) -> dict:
    return {str(k): float(v) for k, v in series.items()}


def _safe_growth(current: float, previous: float) -> float:
    if previous <= 0:
        return 0.0
    return float(((current - previous) / previous) * 100.0)


def _subscription_vs_cash_breakdown(df: pd.DataFrame) -> dict:
    payment = df["type_paiement"].astype(str).str.strip().str.lower()
    subscription_status = df["statut_abonnement"].astype(str).str.strip().str.lower()

    is_subscriber = (
        subscription_status.eq("oui")
        | payment.isin({"abonne", "abonné", "badge"})
    )
    is_cash = payment.isin({"especes", "espèces", "cash"})

    abonnes_revenue = float(df.loc[is_subscriber, "montant_net"].sum())
    cash_revenue = float(df.loc[is_cash, "montant_net"].sum())
    total_revenue = float(df["montant_net"].sum())

    return {
        "Abonnés": abonnes_revenue,
        "Cash": cash_revenue,
        "Autres": float(max(total_revenue - abonnes_revenue - cash_revenue, 0.0)),
    }


def _period_revenue(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> float:
    mask = (df["date_heure"] >= start_ts) & (df["date_heure"] < end_ts)
    return float(df.loc[mask, "montant_net"].sum())


def get_traffic_stats() -> dict:
    df = _load_traffic_data()

    if df.empty:
        return {
            "summary": {
                "total_transactions": 0,
                "total_revenue": 0.0,
                "avg_ticket": 0.0,
                "median_ticket": 0.0,
                "avg_transactions_per_day": 0.0,
                "avg_revenue_per_day": 0.0,
                "active_days": 0,
                "growth_last_7_days_pct": 0.0,
                "growth_last_30_days_pct": 0.0,
            },
            "period": {"start": None, "end": None},
            "peaks": {
                "day": {"date": None, "transactions": 0},
                "hour": {"hour": None, "transactions": 0},
            },
            "top": {"gare": {"name": None, "transactions": 0}},
            "breakdowns": {
                "by_gare": {},
                "by_classe": {},
                "by_payment": {},
                "by_weekday": {},
                "by_hour": {},
            },
            "revenue": {
                "daily": {},
                "by_gare": {},
                "by_classe": {},
                "by_payment": {},
                "abonne_vs_cash": {},
                "best_day": {"date": None, "amount": 0.0},
                "worst_day": {"date": None, "amount": 0.0},
            },
        }

    df["date"] = df["date_heure"].dt.date
    df["hour"] = df["date_heure"].dt.hour

    daily_counts = df.groupby("date")["id_transaction"].count()
    daily_revenue = df.groupby("date")["montant_net"].sum()
    hourly_counts = df.groupby("hour")["id_transaction"].count()
    revenue_by_gare = df.groupby("gare")["montant_net"].sum().sort_values(ascending=False)
    revenue_by_classe = df.groupby("classe_vehicule")["montant_net"].sum().sort_values(ascending=False)
    revenue_by_payment = df.groupby("type_paiement")["montant_net"].sum().sort_values(ascending=False)
    revenue_daily_sorted = daily_revenue.sort_index()
    daily_max_date = revenue_daily_sorted.idxmax()
    daily_min_date = revenue_daily_sorted.idxmin()

    max_ts = df["date_heure"].max()
    end_current_7d = max_ts + pd.Timedelta(hours=1)
    start_current_7d = end_current_7d - pd.Timedelta(days=7)
    start_previous_7d = start_current_7d - pd.Timedelta(days=7)

    end_current_30d = end_current_7d
    start_current_30d = end_current_30d - pd.Timedelta(days=30)
    start_previous_30d = start_current_30d - pd.Timedelta(days=30)

    current_7d = _period_revenue(df, start_current_7d, end_current_7d)
    previous_7d = _period_revenue(df, start_previous_7d, start_current_7d)
    current_30d = _period_revenue(df, start_current_30d, end_current_30d)
    previous_30d = _period_revenue(df, start_previous_30d, start_current_30d)

    weekday_names = {
        0: "Lundi",
        1: "Mardi",
        2: "Mercredi",
        3: "Jeudi",
        4: "Vendredi",
        5: "Samedi",
        6: "Dimanche",
    }
    weekday_counts = (
        df.assign(weekday_num=df["date_heure"].dt.dayofweek)
        .groupby("weekday_num")["id_transaction"]
        .count()
        .reindex(range(7), fill_value=0)
    )

    peak_day_idx = daily_counts.idxmax()
    peak_hour_idx = hourly_counts.idxmax()
    top_gare_series = df.groupby("gare")["id_transaction"].count().sort_values(ascending=False)

    return {
        "summary": {
            "total_transactions": int(len(df)),
            "total_revenue": float(df["montant_net"].sum()),
            "avg_ticket": float(df["montant_net"].mean()),
            "median_ticket": float(df["montant_net"].median()),
            "avg_transactions_per_day": float(daily_counts.mean()),
            "avg_revenue_per_day": float(daily_revenue.mean()),
            "active_days": int(daily_counts.shape[0]),
            "growth_last_7_days_pct": _safe_growth(current_7d, previous_7d),
            "growth_last_30_days_pct": _safe_growth(current_30d, previous_30d),
        },
        "period": {
            "start": df["date_heure"].min().isoformat(),
            "end": df["date_heure"].max().isoformat(),
        },
        "peaks": {
            "day": {
                "date": str(peak_day_idx),
                "transactions": int(daily_counts.loc[peak_day_idx]),
            },
            "hour": {
                "hour": int(peak_hour_idx),
                "transactions": int(hourly_counts.loc[peak_hour_idx]),
            },
        },
        "top": {
            "gare": {
                "name": str(top_gare_series.index[0]),
                "transactions": int(top_gare_series.iloc[0]),
            }
        },
        "breakdowns": {
            "by_gare": _series_to_dict(top_gare_series),
            "by_classe": _series_to_dict(df.groupby("classe_vehicule")["id_transaction"].count().sort_values(ascending=False)),
            "by_payment": _series_to_dict(df.groupby("type_paiement")["id_transaction"].count().sort_values(ascending=False)),
            "by_weekday": {weekday_names[k]: int(v) for k, v in weekday_counts.items()},
            "by_hour": {str(k): int(v) for k, v in hourly_counts.sort_index().items()},
        },
        "revenue": {
            "daily": _series_to_float_dict(revenue_daily_sorted),
            "by_gare": _series_to_float_dict(revenue_by_gare),
            "by_classe": _series_to_float_dict(revenue_by_classe),
            "by_payment": _series_to_float_dict(revenue_by_payment),
            "abonne_vs_cash": _subscription_vs_cash_breakdown(df),
            "best_day": {"date": str(daily_max_date), "amount": float(revenue_daily_sorted.loc[daily_max_date])},
            "worst_day": {"date": str(daily_min_date), "amount": float(revenue_daily_sorted.loc[daily_min_date])},
        },
    }
