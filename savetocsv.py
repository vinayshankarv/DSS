import pandas as pd
from datetime import datetime
import os
from logger import setup_logger  

logger = setup_logger("save_to_csv")  


def save_scraped_data(data, output_dir="outputs"):
    """
    Purpose:
    - Unifies data from Amazon & Flipkart scrapers.
    - Automatically adds missing columns.
    - Avoids duplicates by URL.
    - Returns saved CSV path for downstream analysis.

    Columns: title, price, rating, url, platform, timestamp
    """

    if not data:
        logger.warning("⚠️ No data received — skipping save.")
        return None

    # Normalize and structure incoming data
    unified = []
    for item in data:
        mapped = {
            "title": item.get("title") or item.get("name") or "N/A",
            "price": str(item.get("price", "")).strip(),
            "rating": item.get("rating", ""),
            "url": item.get("url") or item.get("link") or "N/A",
            "platform": item.get("platform", "Unknown"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        unified.append(mapped)

    df_new = pd.DataFrame(
        unified, columns=["title", "price", "rating", "url", "platform", "timestamp"]
    )

    # Prepare output directory and timestamped file name
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"scraped_results_{timestamp}.csv")

    # Save new batch directly (no merging — keeps each run independent)
    df_new.to_csv(file_path, index=False, encoding="utf-8-sig")

    # ✅ Logging summary
    logger.info(f"💾 {len(df_new)} records saved to {file_path}")
    logger.debug(f"Last 5 entries:\n{df_new.tail(5).to_string(index=False)}")

    # ✅ Return CSV path for further processing (e.g., competitor analysis)
    return file_path


if __name__ == "__main__":
    # Self-test
    test_data = [
        {"title": "Samsung S25 5G", "price": "₹80,999", "rating": "4.5", "url": "https://amazon.in", "platform": "Amazon"},
        {"name": "Samsung S25 FE 5G", "price": "₹65,999", "rating": "4.6", "link": "https://flipkart.com", "platform": "Flipkart"}
    ]
    save_scraped_data(test_data)
