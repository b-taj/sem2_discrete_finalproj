import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.pbs.gov.pk"
START_PAGE = "https://www.pbs.gov.pk/price-statistics/"

os.makedirs("data/pdfs", exist_ok=True)


def get_pdf_links():
    print("Fetching PBS index page...")

    r = requests.get(START_PAGE, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    links = set()

    for a in soup.find_all("a"):
        href = a.get("href", "")
        text = a.text.lower()

        # keep only CPI-related PDFs
        if ".pdf" in href.lower() and "cpi" in href.lower():
            full_url = urljoin(BASE_URL, href)
            links.add(full_url)

    print(f"Found {len(links)} PDF links")
    return sorted(list(links))


def download_pdf(url):
    filename = url.split("/")[-1]
    filepath = os.path.join("data/pdfs", filename)

    if os.path.exists(filepath):
        print(f"Already exists: {filename}")
        return

    try:
        r = requests.get(url, timeout=20, stream=True)

        # validation 1: status
        if r.status_code != 200:
            print(f"SKIP (bad status): {url}")
            return

        # validation 2: content-type
        ctype = r.headers.get("Content-Type", "")
        if "pdf" not in ctype.lower():
            print(f"SKIP (not pdf): {url}")
            return

        # validation 3: PDF signature
        if not r.content.startswith(b"%PDF"):
            print(f"SKIP (invalid pdf): {url}")
            return

        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        print(f"Downloaded: {filename}")
        time.sleep(0.5)

    except Exception as e:
        print(f"FAILED: {url} -> {e}")


def main():
    links = get_pdf_links()

    if not links:
        print("No PDFs found. PBS page structure may have changed.")
        return

    for url in links:
        download_pdf(url)


if __name__ == "__main__":
    main()