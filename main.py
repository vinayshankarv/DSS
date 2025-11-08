# main.py
from amazonscraper import search_amazon
from flipkartscraper import scrape_flipkart_prices
from savetocsv import save_scraped_data
from logger import setup_logger

# Import ML pipeline modules
from ml_pipeline.train_model import train_and_save
from ml_pipeline.predict import predict_price


def main():
    logger = setup_logger("main")

    # =========================
    # 1️⃣ User Input
    # =========================
    print("\n=============================")
    print("🛍️  DSS Pricing Intelligence System")
    print("=============================")
    product_query = input("Enter the product name to search: ").strip()

    if not product_query:
        print("❌ No product entered. Exiting.")
        return

    logger.info(f"Starting unified scraping for: {product_query}")

    # =========================
    # 2️⃣ Amazon Scraper
    # =========================
    try:
        logger.info("🛒 Fetching Amazon results...")
        amazon_data = search_amazon(product_query)
        logger.info(f"✅ Amazon: {len(amazon_data)} items scraped.")
    except Exception as e:
        logger.error(f"❌ Amazon scraper failed: {e}")
        amazon_data = []

    # =========================
    # 3️⃣ Flipkart Scraper
    # =========================
    try:
        logger.info("🛍️ Fetching Flipkart results...")
        flipkart_data = scrape_flipkart_prices(product_query)
        logger.info(f"✅ Flipkart: {len(flipkart_data)} items scraped.")
    except Exception as e:
        logger.error(f"❌ Flipkart scraper failed: {e}")
        flipkart_data = []

    # =========================
    # 4️⃣ Combine & Save
    # =========================
    combined_data = amazon_data + flipkart_data
    if not combined_data:
        logger.warning("⚠️ No data scraped from either platform. Exiting.")
        return

    save_scraped_data(combined_data)
    logger.info(f"💾 Data saved successfully. Total entries: {len(combined_data)}")

    # =========================
    # 5️⃣ Train Model
    # =========================
    try:
        logger.info("🤖 Training model on scraped data...")
        train_and_save()
        logger.info("📈 Model training completed and saved.")
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        return

    # =========================
    # 6️⃣ Optional Prediction
    # =========================
    predict_choice = input("\nDo you want to predict a price? (y/n): ").strip().lower()
    if predict_choice == "y":
        title = input("Enter product title (e.g., 'Samsung Galaxy S25 Ultra 5G'): ").strip()
        platform = input("Enter platform (Amazon / Flipkart): ").strip().capitalize() or "Amazon"
        rating = float(input("Enter rating (default 4.5): ") or 4.5)
        timestamp = input("Enter timestamp (YYYY-MM-DD, default today): ").strip()

        sample_record = {
            "title": title,
            "platform": platform,
            "rating": rating,
            "timestamp": timestamp
        }

        try:
            predicted_price = predict_price(sample_record, model_name="rf")
            logger.info(f"💡 Predicted price for '{sample_record['title']}': ₹{predicted_price:.2f}")
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")

    logger.info("✅ DSS pipeline run completed successfully.\n")


if __name__ == "__main__":
    main()
