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
        montant_net
    FROM fact_transactions
    WHERE montant_net > 0
    """
    df = pd.read_sql(query, get_engine())
    df["date_heure"] = pd.to_datetime(df["date_heure"])
    return df


def _series_to_dict(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.items()}


def get_traffic_stats() -> dict:
    df = _load_traffic_data()

    if df.empty:
        return {
            "summary": {
                "total_transactions": 0,
                "total_revenue": 0.0,
                "avg_transactions_per_day": 0.0,
                "avg_revenue_per_day": 0.0,
                "active_days": 0,
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
        }

    df["date"] = df["date_heure"].dt.date
    df["hour"] = df["date_heure"].dt.hour

    daily_counts = df.groupby("date")["id_transaction"].count()
    daily_revenue = df.groupby("date")["montant_net"].sum()
    hourly_counts = df.groupby("hour")["id_transaction"].count()

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
            "avg_transactions_per_day": float(daily_counts.mean()),
            "avg_revenue_per_day": float(daily_revenue.mean()),
            "active_days": int(daily_counts.shape[0]),
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
    }
