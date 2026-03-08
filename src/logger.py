from loguru import logger
import sys
from pathlib import Path

# Configure logger once
log_path = Path("logs/vantage_core.log")
log_path.parent.mkdir(exist_ok=True)

logger.remove()  # remove default handler

# Console output (clean + colored)
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}",
    level="INFO",
    colorize=True
)

# File output (structured + searchable)
logger.add(
    log_path,
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {name}:{function}:{line} | {message}",
    enqueue=True  # thread-safe
)

# Export the logger
__all__ = ["logger"]