from pathlib import Path


class PathHandler:
    BASE_DIR = Path(__file__).parents[1]
    RESOURCES = BASE_DIR / 'resources'
    SRC = BASE_DIR / 'src'
