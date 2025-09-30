from pathlib import Path

from src.db.utility import load_json_file


BASE_DIR = Path(__file__).resolve().parent.parent
PROFESSORS_FILE_PATH = BASE_DIR / "data" / "professors.json"


def get_professor_by_id(id: str) -> dict:
    professors = load_json_file(PROFESSORS_FILE_PATH)
    for professor in professors:
        if professor.get("id") == id:
            return professor
    return None


def list_professors() -> list[dict]:
    professors = load_json_file(PROFESSORS_FILE_PATH)
    return professors if professors else []
