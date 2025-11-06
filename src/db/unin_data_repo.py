from pathlib import Path

from src.db.utility import load_json_file


BASE_DIR = Path(__file__).resolve().parent.parent
UNIN_DATA_FILE_PATH = BASE_DIR / "data" / "unin_data.json"


def list_data() -> list[dict]:
    data = load_json_file(UNIN_DATA_FILE_PATH)
    return data if data else []
