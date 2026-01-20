import csv
import requests

CSV_INPUT = "clients.csv"
CSV_OUTPUT = "clients_without_seller.csv"
SELLERS_URL = "https://www.mgid.com/sellers.json"

def get_seller_ids():
    response = requests.get(SELLERS_URL, timeout=10)
    response.raise_for_status()
    data = response.json()

    # Extract seller_id values into a set for fast lookup
    return {
        str(seller["seller_id"])
        for seller in data.get("sellers", [])
        if "seller_id" in seller
    }

def filter_clients_without_sellers():
    seller_ids = get_seller_ids()

    with open(CSV_INPUT, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        unmatched_rows = [
            row for row in reader
            if str(row.get("id")) not in seller_ids
        ]

    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unmatched_rows)

    print(f"Saved {len(unmatched_rows)} unmatched rows to '{CSV_OUTPUT}'")

if __name__ == "__main__":
    filter_clients_without_sellers()
