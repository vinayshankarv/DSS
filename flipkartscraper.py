import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from logger import setup_logger

logger = setup_logger("flipkart_scraper")

def scrape_flipkart_prices(product_name):
    logger.info(f"Searching Flipkart for '{product_name}'...")

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get("https://www.flipkart.com")

    try:
        close_btn = driver.find_element(By.XPATH, "//button[contains(text(),'✕')]")
        close_btn.click()
    except:
        pass

    search_box = driver.find_element(By.NAME, "q")
    search_box.send_keys(product_name)
    search_box.send_keys(Keys.RETURN)
    time.sleep(5)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    product_cards = soup.select("div.tUxRFH") or soup.select("div._75nlfW") or soup.select("div._1AtVbE")

    if not product_cards:
        logger.warning("⚠️ No products found — Flipkart layout may have changed.")
        driver.quit()
        return []

    results = []
    for card in product_cards[:10]:
        name = card.select_one("a.IRpwTa, a.s1Q9rs, div.KzDlHZ, a.WKTcLC, div._4rR01T")
        price = card.select_one("div.Nx9bqj._4b5DiR, div._30jeq3")
        rating = card.select_one("div.XQDdHH, div._3LWZlK")
        link_tag = card.find("a", href=lambda x: x and "/p/" in x)
        link = f"https://www.flipkart.com{link_tag['href']}" if link_tag else "N/A"

        results.append({
            "name": name.text.strip() if name else "N/A",
            "price": price.text.strip() if price else "N/A",
            "rating": rating.text.strip() if rating else "N/A",
            "link": link,
            "platform": "Flipkart"
        })

    driver.quit()
    logger.info(f"✅ Flipkart scraper extracted {len(results)} valid products.")
    return results
