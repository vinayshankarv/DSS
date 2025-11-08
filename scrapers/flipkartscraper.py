import sys, os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------
# 🔧 Ensure logger.py in project root is importable
# ---------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import setup_logger

logger = setup_logger("flipkart_scraper")

# ---------------------------------------------------------------------
# --- MAIN SCRAPER FUNCTION ---
# ---------------------------------------------------------------------
def scrape_flipkart_prices(product_name):
    logger.info(f"Searching Flipkart for '{product_name}'...")

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get("https://www.flipkart.com")

    # Close login popup if it appears
    try:
        close_btn = driver.find_element(By.XPATH, "//button[contains(text(),'✕')]")
        close_btn.click()
    except Exception:
        pass

    # Search product
    search_box = driver.find_element(By.NAME, "q")
    search_box.send_keys(product_name)
    search_box.send_keys(Keys.RETURN)
    time.sleep(5)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    product_cards = (
        soup.select("div.tUxRFH")
        or soup.select("div._75nlfW")
        or soup.select("div._1AtVbE")
    )

    if not product_cards:
        logger.warning("⚠️ No products found — Flipkart layout may have changed.")
        driver.quit()
        return []

    results = []
    for card in product_cards[:10]:
        title_el = card.select_one("a.IRpwTa, a.s1Q9rs, div.KzDlHZ, a.WKTcLC, div._4rR01T")
        price_el = card.select_one("div.Nx9bqj._4b5DiR, div._30jeq3")
        rating_el = card.select_one("div.XQDdHH, div._3LWZlK")
        link_tag = card.find("a", href=lambda x: x and "/p/" in x)
        link = f"https://www.flipkart.com{link_tag['href']}" if link_tag else "N/A"

        results.append({
            "title": title_el.text.strip() if title_el else "N/A",
            "price": price_el.text.strip() if price_el else "N/A",
            "rating": rating_el.text.strip() if rating_el else "N/A",
            "url": link,
            "platform": "Flipkart"
        })

    driver.quit()
    logger.info(f"✅ Flipkart scraper extracted {len(results)} valid products.")
    return results


# ---------------------------------------------------------------------
# --- MAIN EXECUTION (for direct run or Streamlit subprocess) ---
# ---------------------------------------------------------------------
if __name__ == "__main__":
    keyword = os.getenv("SCRAPE_KEYWORD", "iPhone 16")
    results = scrape_flipkart_prices(keyword)

    if results:
        import pandas as pd
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", f"scraped_results_flipkart_{keyword.replace(' ', '_')}.csv")
        pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"✅ Data saved to {out_path}")
    else:
        logger.warning(f"No results found for {keyword}.")
