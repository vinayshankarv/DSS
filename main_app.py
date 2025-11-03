import streamlit as st
import pandas as pd
import os
import joblib
import plotly.express as px

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "model")

st.set_page_config(page_title="DSS Pricing Intelligence System", layout="wide", page_icon="📊")

# --- HEADER ---
st.title("📊 DSS Pricing Intelligence System")
st.markdown("---")

# --- SIDEBAR NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Dashboard", "Price Prediction", "Analytics"], index=0)

# --- LOADERS ---
@st.cache_data
def load_latest_data():
    """Loads the latest scraped CSV from outputs folder."""
    if not os.path.exists(OUTPUT_DIR):
        return None, "❌ Output directory not found."
    csv_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv")]
    if not csv_files:
        return None, "❌ No scraped data found."
    latest_file = max([os.path.join(OUTPUT_DIR, f) for f in csv_files], key=os.path.getmtime)
    try:
        df = pd.read_csv(latest_file)
        return df, latest_file
    except Exception as e:
        return None, f"⚠️ Error loading file: {e}"

@st.cache_resource
def load_model():
    """Loads trained model (rf_model.pkl)"""
    model_path = os.path.join(MODEL_DIR, "rf_model.pkl")
    if not os.path.exists(model_path):
        return None, "❌ Model file not found."
    try:
        model = joblib.load(model_path)
        return model, None
    except Exception as e:
        return None, f"⚠️ Error loading model: {e}"

# --- DASHBOARD PAGE ---
if page == "Dashboard":
    st.header("📈 Latest Scrape Overview")

    df, info = load_latest_data()
    if df is None:
        st.warning(info)
    else:
        st.success(f"📂 Loaded file: `{os.path.basename(info)}`")
        st.dataframe(df.head(10), use_container_width=True)

        with st.expander("Summary Statistics"):
            st.write(df.describe(include='all'))

        if "price_inr" in df.columns:
            fig = px.histogram(df, x="price_inr", nbins=30, title="Price Distribution (₹)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Column `price_inr` not found in dataset.")

# --- PRICE PREDICTION PAGE ---
elif page == "Price Prediction":
    st.header("💰 Price Prediction Tool")

    model, err = load_model()
    if model is None:
        st.warning(err)
    else:
        st.success("✅ Model loaded successfully.")

        st.markdown("### Enter Product & Competitor Details")
        product_name = st.text_input("Product Name")
        competitor_price = st.number_input("Competitor Price (₹)", min_value=0.0, step=0.01)

        feature_cols = ["feature_1", "feature_2", "feature_3"]
        features = {}
        for col in feature_cols:
            features[col] = st.number_input(f"{col}", value=0.0, step=0.1)

        if st.button("Predict Recommended Price"):
            if not product_name.strip() or competitor_price <= 0:
                st.error("❌ Please enter valid inputs.")
            else:
                try:
                    X_input = pd.DataFrame([features])
                    prediction = model.predict(X_input)[0]
                    diff_percent = ((prediction - competitor_price) / competitor_price) * 100

                    st.success(f"📦 Predicted Price: ₹{prediction:.2f}")
                    st.info(f"Competitor Price: ₹{competitor_price:.2f}")
                    st.write(f"Difference: {diff_percent:.2f}%")

                    if diff_percent > 5:
                        st.warning(f"⚠️ Overpriced: Predicted price is {diff_percent:.1f}% higher.")
                    elif diff_percent < -5:
                        st.error(f"📉 Underpriced: Predicted price is {abs(diff_percent):.1f}% lower.")
                    else:
                        st.success("✅ Optimal Range: Within ±5% of competitor.")
                except Exception as e:
                    st.error(f"⚠️ Prediction failed: {e}")

# --- ANALYTICS PAGE ---
elif page == "Analytics":
    st.header("📊 Analytics Overview")

    df, _ = load_latest_data()
    if df is not None and not df.empty:
        if "source" in df.columns and "price_inr" in df.columns:
            fig = px.box(df, x="source", y="price_inr", color="source", title="Price Distribution by Source")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Columns `source` or `price_inr` not found for visualization.")

        if "product_name" in df.columns and "price_inr" in df.columns:
            fig2 = px.scatter(df, x="product_name", y="price_inr", title="Product-wise Price Variance")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No data available for analytics.")

# --- FOOTER ---
st.markdown("---")
st.caption("Developed by **Vinay Shankar** | DSS Pricing Intelligence System")
