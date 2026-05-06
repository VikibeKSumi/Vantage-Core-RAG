

import requests
from pathlib import Path
from loguru import logger

headers = {"User-Agent": "Mozilla/5.0"}
download_path = Path(__file__).parent.parent / "data" / "raw_docs"

def volume_1():
    for y2, y1 in enumerate(range(2020, 2026, 1), start=21): 
        try:
            response = requests.get(
                url=f"https://www.indiabudget.gov.in/budget{y1}-{y2}/economicsurvey/doc/echapter.pdf",
                headers=headers
            )

            if response.status_code == 200:
                logger.info(f"downloading {y1} - {y2}...")
                with open(download_path/ f"esFY{y1}.pdf", "wb") as f:
                    f.write(response.content)
            else:
                logger.info(f"File not found {y1}-{y2}")
        except Exception as e:
            logger.error(f"url request failed: {e}...")

def volume_2():
    for y2, y1 in enumerate(range(2019, 2026, 1), start=20): 
        try:
            response = requests.get(
                url=f"https://www.indiabudget.gov.in/budget{y1}-{y2}/economicsurvey/doc/echapter_vol2.pdf",
                headers=headers
            )
            if response.status_code == 200:
                logger.info(f"downloading {y1} - {y2}...")
                with open(download_path/ f"esFY{y1}_vol2.pdf", "wb") as f:
                    f.write(response.content)
            else:
                logger.info(f"File not found {y1}-{y2}")
        except Exception as e:
            logger.error(f"Request url failed: {e}...")


if __name__ == "__main__":
    volume_1()
    volume_2()