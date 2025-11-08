import sys, os
import requests
import time
import random
import re
from urllib.parse import quote_plus, urljoin
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------
# 🔧 Ensure logger.py in project root is importable
# ---------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import setup_logger

logger = setup_logger("amazon_scraper")

# ---------------------------------------------------------------------
# --- CONFIGURATION ---
# ---------------------------------------------------------------------
UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]

# ---------------------------------------------------------------------
# --- SESSION BUILDER ---
# ---------------------------------------------------------------------
def build_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(UA_LIST),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    })
    return s

# ---------------------------------------------------------------------
# --- CLEANERS & EXTRACTORS ---
# ---------------------------------------------------------------------
def clean_price(text: str):
    if not text:
        return None
    text = text.replace("₹", "").replace(",", "").strip()
    m = re.search(r"(\d[\d\.]*)", text)
    return float(m.group(1)) if m else None


def extract_title(node):
    for sel in [
        "h2 a span.a-text-normal",
        "span.a-size-medium.a-color-base.a-text-normal",
        "span.a-size-base-plus.a-color-base.a-text-normal",
        "img.s-image",
    ]:
        el = node.select_one(sel)
        if el:
            return el.get("alt") if sel == "img.s-image" else el.get_text(strip=True)
    return "Unknown Product"


def extract_price(node):
    # Try standard selector
    price_el = node.select_one("span.a-price > span.a-offscreen")
    if price_el:
        return clean_price(price_el.get_text())

    # Fallback for alternate layouts
    alt_price = node.select_one("span.a-price-whole") or node.select_one("span[class*='price']")
    if alt_price:
        return clean_price(alt_price.get_text())

    return None


def extract_rating(node):
    rating_el = node.select_one("span[aria-label*='out of 5 stars']") or node.select_one("span.a-icon-alt")
    if not rating_el:
        return None

    text = rating_el.get("aria-label", rating_el.get_text())
    if "star" not in text.lower():
        return None  # skip non-rating text

    m = re.search(r"(\d+(\.\d+)?)", text)
    rating = float(m.group(1)) if m else None
    if rating and 0 < rating <= 5:
        return rating
    return None


def is_sponsored(node):
    if node.select_one("[aria-label='Sponsored']"):
        return True
    for lbl in node.select("span.s-label-popover-default, span.puis-label-popover-default"):
        if "sponsored" in lbl.get_text(strip=True).lower():
            return True
    return False

# ---------------------------------------------------------------------
# --- MAIN SCRAPER FUNCTION ---
# ---------------------------------------------------------------------
def search_amazon(keyword, max_items=15, session=None):
    session = session or build_session()
    url = f"https://www.amazon.in/s?k={quote_plus(keyword)}"
    logger.info(f"Searching Amazon for '{keyword}'...")

    try:
        r = session.get(url, timeout=25)
        r.raise_for_status()
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return []

    soup = BeautifulSoup(r.text, "lxml")
    nodes = soup.select("div.s-main-slot div[data-component-type='s-search-result'][data-asin]")
    logger.info(f"Found {len(nodes)} candidate nodes on Amazon.")

    results = []
    for node in nodes:
        if is_sponsored(node):
            continue

        title = extract_title(node)
        price = extract_price(node)
        rating = extract_rating(node)
        link_el = (
            node.select_one("h2 a.a-link-normal.s-underline-text.s-underline-link-text.s-link-style.a-text-normal")
            or node.select_one("h2 a.a-link-normal")
            or node.select_one("a.a-link-normal.s-no-outline")
            or node.find("a", href=lambda x: x and "/dp/" in x)
        )
        product_url = urljoin("https://www.amazon.in", link_el["href"]) if link_el and link_el.has_attr("href") else None

        results.append({
            "title": title.strip() if title else "N/A",
            "price": price,
            "rating": rating,
            "url": product_url,
            "platform": "Amazon",
        })

        if len(results) >= max_items:
            break

    # Post-cleaning
    for r in results:
        if isinstance(r["title"], str):
            r["title"] = " ".join(r["title"].split())
        if isinstance(r["price"], str):
            r["price"] = clean_price(r["price"])
        if isinstance(r["rating"], str):
            try:
                r["rating"] = float(r["rating"])
            except:
                r["rating"] = None

    logger.info(f"✅ Amazon scraper extracted {len(results)} valid products.")
    return results


# ---------------------------------------------------------------------
# --- MAIN EXECUTION (for direct run or Streamlit subprocess) ---
# ---------------------------------------------------------------------
if __name__ == "__main__":
    keyword = os.getenv("SCRAPE_KEYWORD", "iPhone 16")
    results = search_amazon(keyword, max_items=15)

    if results:
        import pandas as pd
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", f"scraped_results_amazon_{keyword.replace(' ', '_')}.csv")
        pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"✅ Data saved to {out_path}")
    else:
        logger.warning(f"No results found for {keyword}.")
