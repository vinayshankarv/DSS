#streamlit.py

import streamlit as st
import pandas as pd
import os
from utils.savetocsv import save_scraped_data
from scrapers.amazonscraper import search_amazon
from scrapers.flipkartscraper import scrape_flipkart_prices
from models.modeltrainer import train_and_evaluate_models

st.set_page_config(page_title="E-Commerce Pricing DSS", layout="wide")

st.title("📊 E-Commerce Pricing Intelligence DSS")
st.markdown("Integrated decision support system for automated price intelligence and prediction")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🛒 Scraping & Data", "📈 Price Analytics", "🤖 Price Prediction"])

# ---------------- TAB 1 ----------------
with tab1:
    st.header("Scraping & Data Collection")
    product_query = st.text_input("Enter product keyword (e.g., Samsung S25 5G):")

    if st.button("Scrape Amazon + Flipkart"):
        with st.spinner("Scraping in progress..."):
            all_data = []

            try:
                amazon_data = search_amazon(product_query)
                all_data.extend(amazon_data)
            except Exception as e:
                st.error(f"Amazon scraper failed: {e}")

            try:
                flipkart_data = scrape_flipkart_prices(product_query)
                all_data.extend(flipkart_data)
            except Exception as e:
                st.error(f"Flipkart scraper failed: {e}")

            if all_data:
                save_scraped_data(all_data)
                st.success(f"✅ {len(all_data)} new records scraped and saved.")
            else:
                st.warning("No data retrieved from scrapers.")

    st.divider()
    if os.path.exists("outputs/scraped_results.csv"):
        df = pd.read_csv("outputs/scraped_results.csv")
        st.dataframe(df.tail(10), use_container_width=True)
        st.download_button("⬇️ Download Full Dataset", data=df.to_csv(index=False).encode("utf-8"), file_name="scraped_results.csv")
    else:
        st.info("No data available yet. Scrape to generate results.")

# ---------------- TAB 2 ----------------
with tab2:
    st.header("Competitor Pricing Analytics")
    if os.path.exists("outputs/scraped_results.csv"):
        df = pd.read_csv("outputs/scraped_results.csv")
        st.metric("Total Entries", len(df))
        st.metric("Unique Products", df["title"].nunique())

        st.subheader("Average Price by Platform")
        avg_price = df.groupby("platform")["price"].apply(lambda x: pd.to_numeric(x.str.replace("₹", "").str.replace(",", ""), errors='coerce').mean())
        st.bar_chart(avg_price)

        st.subheader("Top 5 Cheapest Products")
        df["price_num"] = pd.to_numeric(df["price"].str.replace("₹", "").str.replace(",", ""), errors="coerce")
        cheapest = df.sort_values("price_num").head(5)
        st.dataframe(cheapest[["title", "platform", "price", "rating", "url"]])
    else:
        st.warning("Please scrape data first.")

# ---------------- TAB 3 ----------------
with tab3:
    st.header("Price Prediction Models")

    if st.button("Train Models"):
        if os.path.exists("outputs/scraped_results.csv"):
            df = pd.read_csv("outputs/scraped_results.csv")
            results = train_and_evaluate_models(df)
            st.success("Model training completed.")
            st.dataframe(results)
        else:
            st.error("No data found. Please scrape first.")

    if os.path.exists("models/model_performance.csv"):
        st.subheader("Model Performance Overview")
        perf = pd.read_csv("models/model_performance.csv")
        st.dataframe(perf)
    else:
        st.info("Train models to view performance metrics.")
