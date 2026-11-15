import pandas as pd


def select_orders(df, countries, low, high):
    df = df[df["country"].isin(countries)]
    mask = ((df["amount"].between(low, high)) | df["vip"] == True)
    df = df[mask]
    keep = [c for c in df.columns if c != "vip"]
    return df[keep]