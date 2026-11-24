import requests

url = "https://api.fda.gov/drug/enforcement.json?limit=3"
response = requests.get(url)
data = response.json()

for item in data["results"]:
    print("Drug:", item.get("product_description")[:80])
    print("Batch/Lot:", item.get("code_info"))
    print("Reason:", item.get("reason_for_recall"))
    print("Date:", item.get("recall_initiation_date"))
    print("-" * 50)