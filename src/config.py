# src/config.py
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv
import os
        
class Config:
    """Centralized, validated configuration loader."""

    def __init__(self):
        self._path = Path("config/settings.yaml")
        if not self._path.exists():
            raise FileNotFoundError(f"❌ Config file not found: {self._path}")

        with open(self._path, "r", encoding="utf-8") as f:
            self.data: Dict[str, Any] = yaml.safe_load(f)

        self._validate()

        # Centralize Groq API key (no more passing separately)
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("❌ GROQ_API_KEY not found in .env file")
        
    def _validate(self):
        """Basic validation — easy to extend later."""
        required = ["models", "database", "ingestion"]
        for key in required:
            if key not in self.data:
                raise ValueError(f"❌ Missing required section '{key}' in settings.yaml")

        if "embedding" not in self.data["models"]:
            raise ValueError("❌ Missing 'models.embedding' in config")

    # Convenience properties (clean dot access)
    @property
    def models(self) -> Dict:
        return self.data["models"]

    @property
    def database(self) -> Dict:
        return self.data["database"]

    @property
    def ingestion(self) -> Dict:
        return self.data["ingestion"]