from pathlib import Path


# Check if base directory exists, create otherwise
DEFAULT_PATH = Path.home() / ".brainglobe"
DEFAULT_PATH.mkdir(exist_ok=True)
