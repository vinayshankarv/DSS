# ==============================================================
# Scrapwise DSS Pricing Intelligence System — Final Version
# Streamlit UI | Intelligent Scraper | Unified Prediction
# ==============================================================

import os, glob, subprocess, traceback
from datetime import datetime
import numpy as np, pandas as pd, streamlit as st, joblib
import plotly.express as px
from fpdf import FPDF
import matplotlib.pyplot as plt
from ml_pipeline.predict import predict_price

# ==============================================================
# PAGE CONFIG & THEME
# ==============================================================
st.set_page_config(page_title="Scrapwise DSS Pricing Intelligence",
                   layout="wide", page_icon="📊")
st.markdown("""
<style>
:root { --bg:#000;--panel:rgba(20,20,25,0.55);--text:#f2f3f4;--neon:#00f2ff;--accent:#76ff7a;}
html,body,[class*="css"]{background:var(--bg)!important;color:var(--text)!important;
font-family:'Inter',system-ui,sans-serif;}
div.stTabs [data-baseweb="tab"]{background:var(--panel);border-radius:14px;
border:1px solid rgba(255,255,255,0.1);padding:10px 16px;color:var(--text)!important;}
[data-testid="stSidebar"]{background:#07080a;border-right:1px solid rgba(255,255,255,0.08);}
hr{border:none;height:1px;background:linear-gradient(90deg,rgba(255,255,255,0),var(--neon),
rgba(255,255,255,0));margin:8px 0 18px;}
</style>
""", unsafe_allow_html=True)
px.defaults.template = "plotly_dark"

st.title("⚡ Scrapwise DSS Pricing Intelligence System")
st.caption("AI-Powered Market Price Insights · Scrape • Predict • Compare • Analyze • Export")
st.divider()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR, MODEL_DIR, SCRAPERS_DIR = (os.path.join(BASE_DIR, x) for x in
                                       ("outputs", "model", "scrapers"))
for d in [OUTPUT_DIR, MODEL_DIR]: os.makedirs(d, exist_ok=True)

# ==============================================================
# UTILITIES
# ==============================================================
def to_numeric_price(series): 
    s = series.astype(str).str.replace("₹","").str.replace(",","")
    return pd.to_numeric(s.str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
def format_currency(x): 
    try: return f"₹{float(x):,.0f}"
    except: return "—"

@st.cache_resource
def load_model():
    path = os.path.join(MODEL_DIR,"rf_model.pkl")
    if not os.path.exists(path): return None,"❌ rf_model.pkl missing in /model/"
    try: return joblib.load(path),None
    except Exception as e: return None,f"⚠️ Error loading model: {e}"
model, model_err = load_model()

# ==============================================================
# SCRAPER FUNCTION
# ==============================================================
def run_scraper(keyword:str, platform:str):
    st.info(f"🔎 Scraping for **{keyword}** on **{platform}**…")
    amz, flp = [os.path.join(SCRAPERS_DIR,x) for x in ("amazonscraper.py","flipkartscraper.py")]
    if platform in ("Amazon","Both") and os.path.exists(amz):
        subprocess.run(["python",amz],cwd=BASE_DIR,check=False)
    if platform in ("Flipkart","Both") and os.path.exists(flp):
        subprocess.run(["python",flp],cwd=BASE_DIR,check=False)
    amz_files, flp_files = glob.glob(f"{OUTPUT_DIR}/*amazon*.csv"), glob.glob(f"{OUTPUT_DIR}/*flipkart*.csv")

    if platform=="Both" and amz_files and flp_files:
        a, f = pd.read_csv(max(amz_files,key=os.path.getmtime)), pd.read_csv(max(flp_files,key=os.path.getmtime))
        a["platform"], f["platform"] = "Amazon","Flipkart"
        merged = pd.concat([a,f],ignore_index=True)
        merged["price_num"]=to_numeric_price(merged.get("price",merged.iloc[:,1]))
        merged.drop_duplicates(subset=["title","platform"],inplace=True)
        fname=os.path.join(OUTPUT_DIR,f"scraped_results_combined_{keyword.replace(' ','_')}_{datetime.now():%Y%m%d_%H%M%S}.csv")
        merged.to_csv(fname,index=False,encoding="utf-8-sig")
        return fname
    elif platform=="Amazon" and amz_files:
        return max(amz_files,key=os.path.getmtime)
    elif platform=="Flipkart" and flp_files:
        return max(flp_files,key=os.path.getmtime)
    else: return None

# ==============================================================
# SMART FILTER — MAIN vs VARIANTS
# ==============================================================
def split_main_and_related(df, keyword):
    if df.empty: return pd.DataFrame(),pd.DataFrame()
    t = df["title"].astype(str).str.lower().replace(r"[\(\)\[\],\-]", " ", regex=True).str.replace(r"\s+"," ",regex=True)
    key = keyword.lower().strip().replace("gb"," gb")
    base = key.replace(" ",r"\s*[\(\)\-\,]*\s*")
    main_pat = rf"(apple\s*)?{base}(?!\s*(e|plus|pro|max|ultra|mini|se|air))"
    rel_pat  = rf"{base}\s*(e|plus|pro|max|ultra|mini|se|air)"
    main, rel = df[t.str.contains(main_pat,regex=True)], df[t.str.contains(rel_pat,regex=True)]
    for d in [main,rel]:
        if not d.empty:
            d["rounded_price"]=d["price_num"].round(-2)
            d.drop_duplicates(subset=["platform","rounded_price"],inplace=True)
    return main,rel

# ==============================================================
# SIDEBAR
# ==============================================================
st.sidebar.markdown("### 🔁 Refresh Cache")
if st.sidebar.button("♻️ Clear Cache"):
    st.cache_data.clear(); st.cache_resource.clear(); st.success("Cache cleared.")

# ==============================================================
# TABS
# ==============================================================
tab_scraper, tab_predict, tab_reports = st.tabs(["🛒 Scraper","🤖 Prediction","🧠 Reports"])

# ==============================================================
# TAB 1 — SCRAPER
# ==============================================================
with tab_scraper:
    st.subheader("🛒 Intelligent Product Scraper")
    c1,c2 = st.columns([2,1])
    keyword=c1.text_input("Enter Product Keyword","iPhone 16 128 GB")
    platform=c2.selectbox("Select Platform",["Amazon","Flipkart","Both"],index=2)

    if st.button("🚀 Run Scraper",use_container_width=True):
        path = run_scraper(keyword,platform)
        if path and os.path.exists(path):
            df=pd.read_csv(path); df["price_num"]=to_numeric_price(df.get("price",df.iloc[:,1])); df.dropna(subset=["price_num"],inplace=True)
            main_df,rel_df = split_main_and_related(df,keyword)
            tot, a, f = len(df), len(df[df["platform"].str.contains("amazon",case=False,na=False)]), len(df[df["platform"].str.contains("flipkart",case=False,na=False)])
            st.success(f"✅ Merged Amazon + Flipkart → {os.path.basename(path)} ({tot} total)")
            c1,c2,c3=st.columns(3); c1.metric("Total Relevant",tot); c2.metric("Amazon",a); c3.metric("Flipkart",f)

            # --- main matches
            if not main_df.empty:
                st.markdown("### 🔍 Matched Product Listings")
                for _,r in main_df.iterrows():
                    st.markdown(f"""
                    <div style='background:rgba(20,20,25,0.55);padding:12px 16px;
                    border-radius:12px;margin-bottom:8px;border:1px solid rgba(255,255,255,0.05);'>
                    <span style='background:rgba(255,255,255,0.08);padding:3px 8px;
                    border-radius:8px;font-size:12px;'>{r['platform']}</span>
                    <b style='margin-left:6px;color:#f2f3f4;font-size:15px;'>{r['title'][:160]}...</b><br>
                    <span style='color:#ffcc00;'>💰 Market: ₹{r['price_num']:,.0f}</span><br>
                    <a href='{r['url']}' target='_blank' style='color:#00f2ff;text-decoration:none;'>Open</a></div>
                    """,unsafe_allow_html=True)
            # --- variants
            if not rel_df.empty:
                st.markdown("### 💡 Related Model Variants")
                for _,r in rel_df.iterrows():
                    st.markdown(f"""
                    <div style='background:rgba(20,20,25,0.55);padding:12px 16px;
                    border-radius:12px;margin-bottom:8px;border:1px solid rgba(255,255,255,0.05);'>
                    <span style='background:rgba(255,255,255,0.08);padding:3px 8px;
                    border-radius:8px;font-size:12px;'>{r['platform']}</span>
                    <b style='margin-left:6px;color:#f2f3f4;font-size:15px;'>{r['title'][:160]}...</b><br>
                    <span style='color:#ffcc00;'>💰 Market: ₹{r['price_num']:,.0f}</span><br>
                    <a href='{r['url']}' target='_blank' style='color:#00f2ff;text-decoration:none;'>Open</a></div>
                    """,unsafe_allow_html=True)

# ==============================================================
# TAB 2 — PREDICTION
# ==============================================================
with tab_predict:
    st.header("🤖 Predict Product Price (Auto Mode)")
    if model_err: st.error(model_err)
    elif model is None: st.warning("⚠️ Model not loaded.")
    else:
        files=sorted(glob.glob(os.path.join(OUTPUT_DIR,"*.csv")),key=os.path.getmtime,reverse=True)
        if not files: st.warning("No scraped data found.")
        else:
            f=files[0]; df=pd.read_csv(f); df["price_num"]=to_numeric_price(df.get("price",df.iloc[:,1]))
            st.caption(f"📦 Using latest scraped file: {os.path.basename(f)}")
            if "platform" not in df.columns: df["platform"]="Unknown"
            df=df.dropna(subset=["price_num"])
            try:
                df["Predicted_Price"]=df.apply(lambda r: predict_price(r.to_dict(),"rf"),axis=1)
                df["Price_Gap"]=df["price_num"]-df["Predicted_Price"]
                df["Status"]=np.where(abs(df["Price_Gap"])<500,"Fairly Priced",
                                      np.where(df["Price_Gap"]>0,"Overpriced","Underpriced"))
                comp=df.groupby("platform")[["price_num","Predicted_Price"]].mean().reset_index()
                st.markdown("### 📊 Platform-wise Pricing Comparison")
                st.dataframe(comp.style.format({"price_num":"₹{:.0f}","Predicted_Price":"₹{:.0f}"}),use_container_width=True)
                st.plotly_chart(px.bar(comp,x="platform",y=["price_num","Predicted_Price"],barmode="group",
                                       title="Market vs Predicted Price"),use_container_width=True)
            except Exception as e: st.error(f"Prediction failed: {e}")

# ==============================================================
# TAB 3 — REPORTS
# ==============================================================
with tab_reports:
    st.subheader("🧠 Reports / Insights")
    files=sorted(glob.glob(os.path.join(OUTPUT_DIR,"*.csv")),key=os.path.getmtime,reverse=True)
    if not files: st.info("No data yet.")
    else:
        df=pd.read_csv(files[0]); df["price_num"]=to_numeric_price(df.get("price",df.iloc[:,1]))
        if {"price_num","Predicted_Price"}.issubset(df.columns):
            st.plotly_chart(px.scatter(df,x="price_num",y="Predicted_Price",trendline="ols",
                                       title="Actual vs Predicted"),use_container_width=True)
        st.dataframe(df.head(30),use_container_width=True)

st.markdown("<hr/>",unsafe_allow_html=True)
st.caption("© Scrapwise DSS Pricing Intelligence | Streamlit")
