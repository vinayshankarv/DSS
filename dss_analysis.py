import pandas as pd
import numpy as np
import os
from datetime import datetime

def analyze_latest_scrape():
    folder = "outputs"
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not csv_files:
        print("❌ No scraped data files found.")
        return

    latest_file = max(csv_files, key=lambda f: os.path.getmtime(os.path.join(folder, f)))
    file_path = os.path.join(folder, latest_file)
    print(f"📂 Using latest scraped file: {file_path}")

    df = pd.read_csv(file_path)

    # --- Clean and convert prices ---
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"].notnull() & (df["price"] > 0)]

    if df.empty:
        raise ValueError("No valid price data found after cleaning. Check your scraped data.")

    # --- Market metrics ---
    prices = df["price"].values
    mean_p, std_p = np.mean(prices), np.std(prices)
    low_p, high_p = np.min(prices), np.max(prices)
    median_p = np.median(prices)

    amazon_mean = df[df["platform"].str.lower().str.contains("amazon", na=False)]["price"].mean()
    flip_mean = df[df["platform"].str.lower().str.contains("flipkart", na=False)]["price"].mean()

    # --- Derived metrics ---
    rec_penetration = mean_p - 0.5 * std_p
    rec_competitive = mean_p
    rec_premium = mean_p + 0.5 * std_p

    confidence = max(0, 1 - (std_p / mean_p)) if mean_p > 0 else 0
    volatility = (high_p - low_p) / mean_p * 100 if mean_p > 0 else 0
    platform_bias = abs(amazon_mean - flip_mean) / mean_p * 100 if mean_p > 0 else 0

    # --- Safe print helper ---
    def safe_price(p):
        return f"₹{p:,.0f}" if pd.notna(p) else "N/A"

    # --- Print report ---
    print("\n📊 MARKET PRICE INTELLIGENCE REPORT")
    print("─────────────────────────────────────")
    print(f"Min Price: {safe_price(low_p)}")
    print(f"Max Price: {safe_price(high_p)}")
    print(f"Mean Price: {safe_price(mean_p)}")
    print(f"Median Price: {safe_price(median_p)}")
    print(f"Std Deviation: {safe_price(std_p)}")
    print(f"Volatility Index: {volatility:.2f}%")
    print(f"Platform Bias: {platform_bias:.2f}%")
    print(f"Confidence Index: {confidence*100:.1f}%")

    print("\n💡 PRICE RECOMMENDATIONS")
    print("─────────────────────────────────────")
    print(f"Penetration Strategy: {safe_price(rec_penetration)}")
    print(f"Competitive Strategy: {safe_price(rec_competitive)}")
    print(f"Premium Strategy: {safe_price(rec_premium)}")

    print("\n⚙️ PLATFORM AVERAGES")
    print(f"Amazon Mean: {safe_price(amazon_mean)}")
    print(f"Flipkart Mean: {safe_price(flip_mean)}")

    # --- Save report ---
    out_path = os.path.join("outputs", "pricing_analysis_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Pricing Intelligence Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"Source: {latest_file}\n\n")
        f.write(f"Min Price: {safe_price(low_p)}\n")
        f.write(f"Max Price: {safe_price(high_p)}\n")
        f.write(f"Mean Price: {safe_price(mean_p)}\n")
        f.write(f"Median Price: {safe_price(median_p)}\n")
        f.write(f"Std Deviation: {safe_price(std_p)}\n")
        f.write(f"Volatility Index: {volatility:.2f}%\n")
        f.write(f"Platform Bias: {platform_bias:.2f}%\n")
        f.write(f"Confidence Index: {confidence*100:.1f}%\n\n")
        f.write("Price Recommendations:\n")
        f.write(f"  Penetration Strategy: {safe_price(rec_penetration)}\n")
        f.write(f"  Competitive Strategy: {safe_price(rec_competitive)}\n")
        f.write(f"  Premium Strategy: {safe_price(rec_premium)}\n\n")
        f.write("Platform Averages:\n")
        f.write(f"  Amazon Mean: {safe_price(amazon_mean)}\n")
        f.write(f"  Flipkart Mean: {safe_price(flip_mean)}\n")

    print(f"\n🧾 Report saved to: {out_path}")

if __name__ == "__main__":
    analyze_latest_scrape()
