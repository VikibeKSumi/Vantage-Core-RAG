import requests
from pathlib import Path
from loguru import logger

download_path = Path(__file__).parent.parent / "data" / "raw_docs"

def budget_speech_download():
    try:
      
        for y2, y1 in enumerate(range(2019, 2026), start=20):

            urls =[
                f"https://www.indiabudget.gov.in/doc/bspeech/bs{y1}_{y2}.pdf",
                f"https://www.indiabudget.gov.in/doc/bspeech/bs{y1}{y2}.pdf"
            ]
            for url in urls:
                response = requests.get(
                    url=url
                )
                if response.status_code == 200:
                    logger.info(f"downloading {y1}-{y2}....")
                    with open(download_path/f"bsFY{y2}.pdf", "wb") as f:
                        f.write(response.content)
                    break
                else:
                    logger.info(f"{y1}-{y2} File not found....")

    except Exception as e:
        logger.error(f"Request url failed: {e}....")


    

if __name__ == "__main__":
    budget_speech_download()

        
