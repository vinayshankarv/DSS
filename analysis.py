# ml_pipeline/analysis.py
import pandas as pd

def analyze_competitor_prices(csv_path="outputs/scraped_results.csv", output_path="outputs/competitor_analysis.csv"):
    df = pd.read_csv(csv_path)
    df["price"] = pd.to_numeric(df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    df["normalized_title"] = (
        df["title"].fillna(df.get("name","")).astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # pivot mean price per platform
    summary = df.pivot_table(index="normalized_title", columns="platform", values="price", aggfunc="mean").reset_index()
    # compute difference columns
    def compare(row):
        a = row.get("Amazon")
        f = row.get("Flipkart")
        if pd.isna(a) or pd.isna(f):
            return "Incomplete data"
        if a < f:
            diff = ((f - a) / f) * 100
            return f"Amazon cheaper by {diff:.2f}%"
        elif f < a:
            diff = ((a - f) / a) * 100
            return f"Flipkart cheaper by {diff:.2f}%"
        else:
            return "Same price"

    summary["Comparison"] = summary.apply(compare, axis=1)
    latest = df.groupby("normalized_title")["title"].agg(lambda x: x.dropna().iloc[-1] if len(x.dropna()) else "").reset_index()
    out = summary.merge(latest, on="normalized_title", how="left")
    out = out.rename(columns={"title": "representative_title"})
    out.to_csv(output_path, index=False)
    return out
