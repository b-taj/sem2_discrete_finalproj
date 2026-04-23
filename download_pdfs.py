import requests
import os
import time

os.makedirs("data/pdfs", exist_ok=True)

# URL patterns PBS uses (two slightly different formats)
patterns = [
    "https://www.pbs.gov.pk/sites/default/files/price_statistics/cpi/CPI_Review_{month}_{year}.pdf",
    "https://www.pbs.gov.pk/sites/default/files/price_statistics/cpi/CPI_Monthly_Review_{month}_{year}.pdf",
]

months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

month_num = {m: i+1 for i, m in enumerate(months)}

# Download 3 years: 2022, 2023, 2024
for year in [2022, 2023, 2024]:
    for month in months:
        saved = False
        for pattern in patterns:
            url = pattern.format(month=month, year=year)
            filename = f"data/pdfs/{year}_{month_num[month]:02d}.pdf"

            if os.path.exists(filename):
                print(f"Already exists: {filename}")
                saved = True
                break

            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    with open(filename, "wb") as f:
                        f.write(r.content)
                    print(f"Downloaded: {filename}")
                    saved = True
                    time.sleep(0.5)  # be polite to the server
                    break
            except Exception as e:
                print(f"  Failed {url}: {e}")

        if not saved:
            print(f"  NOT FOUND: {year} {month}")