from pathlib import Path
from typing import Any, Dict
import yaml
import os
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class Config:
    """centralized and validated configuration"""

    def __init__(self):
        self.path = Path(__file__).parent / "settings.yaml"


        if not self.path.exists():
            logger.error(f"Settings file not found in {self.path}")
            raise SystemExit(1)

        with open(self.path, "r", encoding="utf-8") as f:
            self.data: Dict[str, Any] = yaml.safe_load(f)

        self.validate()

        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.error(f"GROQ_API_KEY not found in .env file")
            raise SystemExit(1)
        
    def validate(self):
        
        required = ["models", "database"]
        for key in required:
            if key not in self.data:
                logger.error(f"Missing!! Required section '{key}' in settings.yaml")
                raise SystemExit(1)


    # Convenience properties
    @property
    def models(self) -> Dict:
        return self.data["models"]

    @property
    def database(self) -> Dict:
        return self.data["database"]

    @property
    def ingestion(self) -> Dict:
        return self.data["ingestion"]
    

config = Config()