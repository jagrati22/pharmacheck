from playwright.sync_api import sync_playwright

url = "https://cdscoonline.gov.in/CDSCO/viewPublicNSQDrug"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_context(ignore_https_errors=True).new_page()
    
    print("Opening CDSCO Public NSQ Portal...")
    page.goto(url, timeout=40000)
    page.wait_for_timeout(5000)
    
    rows = page.query_selector_all("table tr")
    print(f"Total NSQ records found on page: {len(rows)}\n")
    
    for r in rows[:6]:
        cols = [c.inner_text().strip().replace("\n", " ") for c in r.query_selector_all("th, td")]
        if cols:
            print(cols)
            print("-" * 60)
            
    browser.close()